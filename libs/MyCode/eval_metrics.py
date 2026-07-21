"""
Evaluation metrics for synthetic time series — aligned with TimeGAN baseline protocol.

- Discriminative Score: Can a post-hoc GRU distinguish real vs synthetic? (lower = better)
- Predictive Score: Train on synthetic, test one-step-ahead on real (lower = better)

Protocol follows: Yoon et al., "Time-series GANs", NeurIPS 2019.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import accuracy_score, mean_absolute_error
import warnings
warnings.filterwarnings('ignore')


# ---------------------------------------------------------------------------
# Model definitions
# ---------------------------------------------------------------------------

class _Discriminator(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self.gru = nn.GRU(input_dim, hidden_dim, num_layers=1, batch_first=True)
        self.fc = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        _, h_n = self.gru(x)
        logit = self.fc(h_n[-1])  # (batch, 1)
        return logit, torch.sigmoid(logit)


class _Predictor(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self.gru = nn.GRU(input_dim, hidden_dim, num_layers=1, batch_first=True)
        self.fc = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        outputs, _ = self.gru(x)          # (batch, seq_len, hidden_dim)
        logit = self.fc(outputs)          # (batch, seq_len, 1)
        return torch.sigmoid(logit)


# ---------------------------------------------------------------------------
# Discriminative Score
# ---------------------------------------------------------------------------

def discriminative_score(real_data, fake_data, iterations=2000, device='cuda', batch_size=128):
    """
    Post-hoc GRU discriminator trained to separate real vs synthetic.

    Args:
        real_data: (N, seq_len, features)
        fake_data: (N, seq_len, features)
        iterations: training iterations (2000, matching baselines)
        device: 'cuda' or 'cpu'
        batch_size: mini-batch size

    Returns:
        disc_score: |accuracy - 0.5|  (0 = perfectly indistinguishable)
    """
    real_data = np.asarray(real_data)
    fake_data = np.asarray(fake_data)

    no, seq_len, dim = real_data.shape
    hidden_dim = max(int(dim / 2), 1)

    # Train/test split — real and fake separated (matching baseline protocol)
    def _split(data, rate=0.8):
        n = len(data)
        idx = np.random.permutation(n)
        split = int(n * rate)
        return data[idx[:split]], data[idx[split:]]

    train_real, test_real = _split(real_data)
    train_fake, test_fake = _split(fake_data)

    to_t = lambda a: torch.FloatTensor(a).to(device)
    train_real_t, test_real_t = to_t(train_real), to_t(test_real)
    train_fake_t, test_fake_t = to_t(train_fake), to_t(test_fake)

    model = _Discriminator(dim, hidden_dim).to(device)
    optimizer = optim.Adam(model.parameters())
    criterion = nn.BCEWithLogitsLoss()

    model.train()
    for _ in range(iterations):
        idx_r = torch.randperm(len(train_real_t))[:batch_size]
        idx_f = torch.randperm(len(train_fake_t))[:batch_size]

        X_real = train_real_t[idx_r]
        X_fake = train_fake_t[idx_f]

        logit_real, _ = model(X_real)
        logit_fake, _ = model(X_fake)

        loss = criterion(logit_real, torch.ones_like(logit_real)) + \
               criterion(logit_fake, torch.zeros_like(logit_fake))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    model.eval()
    with torch.no_grad():
        _, y_pred_real = model(test_real_t)
        _, y_pred_fake = model(test_fake_t)

    y_pred_real = y_pred_real.squeeze().cpu().numpy()
    y_pred_fake = y_pred_fake.squeeze().cpu().numpy()

    # Guard against 0-d tensors when test split has 1 sample
    y_pred_real = np.atleast_1d(y_pred_real)
    y_pred_fake = np.atleast_1d(y_pred_fake)

    y_pred_final = np.concatenate([y_pred_real, y_pred_fake])
    y_label_final = np.concatenate([np.ones(len(y_pred_real)), np.zeros(len(y_pred_fake))])

    acc = accuracy_score(y_label_final, (y_pred_final > 0.5))
    disc_score = float(np.abs(0.5 - acc))

    return disc_score, float(acc)


def discriminative_per_window(real_data, fake_data, iterations=2000,
                              device='cuda', batch_size=128, seed=None):
    """
    Same post-hoc GRU discriminator as `discriminative_score`, but returns a
    per-window verdict for EVERY real and fake window (not just the test split).

    The discriminator is still trained only on the 80% train split (so the
    protocol/score is unchanged); afterwards it is run over ALL windows so you
    can inspect how each one is classified. The `split` column tells you whether
    a window was used to train the discriminator ('train') or held out ('test');
    treat 'test' rows as the honest ones — 'train' rows are optimistic because
    the discriminator has already seen them.

    Returns:
        table: pandas.DataFrame (one row per window) with columns
            kind        : 'real' | 'fake'
            index       : row index into real_data / fake_data
            split       : 'train' | 'test'
            prob_real   : P(real) the discriminator assigned in [0, 1]
            pred        : 'real' | 'fake'  (prob_real > 0.5)
            correct     : bool, whether the guess matched the true label
        disc_score, acc : the usual scalars computed on the test split
    """
    import pandas as pd

    if seed is not None:
        np.random.seed(seed)
        torch.manual_seed(seed)

    real_data = np.asarray(real_data)
    fake_data = np.asarray(fake_data)

    no, seq_len, dim = real_data.shape
    hidden_dim = max(int(dim / 2), 1)

    # Split real and fake separately, KEEPING the shuffle indices this time.
    def _split_idx(n, rate=0.8):
        idx = np.random.permutation(n)
        cut = int(n * rate)
        return idx[:cut], idx[cut:]

    train_idx_r, test_idx_r = _split_idx(len(real_data))
    train_idx_f, test_idx_f = _split_idx(len(fake_data))

    to_t = lambda a: torch.FloatTensor(a).to(device)
    train_real_t = to_t(real_data[train_idx_r])
    train_fake_t = to_t(fake_data[train_idx_f])

    model = _Discriminator(dim, hidden_dim).to(device)
    optimizer = optim.Adam(model.parameters())
    criterion = nn.BCEWithLogitsLoss()

    model.train()
    for _ in range(iterations):
        idx_r = torch.randperm(len(train_real_t))[:batch_size]
        idx_f = torch.randperm(len(train_fake_t))[:batch_size]
        logit_real, _ = model(train_real_t[idx_r])
        logit_fake, _ = model(train_fake_t[idx_f])
        loss = criterion(logit_real, torch.ones_like(logit_real)) + \
               criterion(logit_fake, torch.zeros_like(logit_fake))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    # Predict on EVERY window (both real and fake, both splits).
    model.eval()
    with torch.no_grad():
        _, prob_real = model(to_t(real_data))
        _, prob_fake = model(to_t(fake_data))
    prob_real = np.atleast_1d(prob_real.squeeze().cpu().numpy())
    prob_fake = np.atleast_1d(prob_fake.squeeze().cpu().numpy())

    test_set_r = set(test_idx_r.tolist())
    test_set_f = set(test_idx_f.tolist())

    rows = []
    for i, p in enumerate(prob_real):          # true label = real (1)
        rows.append({
            "kind": "real", "index": i,
            "split": "test" if i in test_set_r else "train",
            "prob_real": float(p), "pred": "real" if p > 0.5 else "fake",
            "correct": bool(p > 0.5),
        })
    for i, p in enumerate(prob_fake):          # true label = fake (0)
        rows.append({
            "kind": "fake", "index": i,
            "split": "test" if i in test_set_f else "train",
            "prob_real": float(p), "pred": "real" if p > 0.5 else "fake",
            "correct": bool(p <= 0.5),
        })
    table = pd.DataFrame(rows)

    # Recompute the usual scalars on the TEST split so it matches the metric.
    tp = np.concatenate([prob_real[test_idx_r], prob_fake[test_idx_f]])
    tl = np.concatenate([np.ones(len(test_idx_r)), np.zeros(len(test_idx_f))])
    acc = accuracy_score(tl, tp > 0.5)
    disc_score = float(np.abs(0.5 - acc))

    return table, disc_score, float(acc)


# ---------------------------------------------------------------------------
# Predictive Score
# ---------------------------------------------------------------------------

def predictive_score(real_data, fake_data, iterations=5000, device='cuda', batch_size=128):
    """
    Post-hoc GRU trained on synthetic data for one-step-ahead prediction of the
    last feature; evaluated on real data. MAE averaged over samples.

    Input:  data[:, :-1, :-1]  (all timesteps except last, all features except last)
    Target: data[:, 1:, -1:]   (all timesteps except first, last feature only)

    Args:
        real_data: (N, seq_len, features)
        fake_data: (N, seq_len, features)
        iterations: training iterations (5000, matching baselines)
        device: 'cuda' or 'cpu'
        batch_size: mini-batch size

    Returns:
        predictive_score: MAE / num_samples
    """
    real_data = np.asarray(real_data)
    fake_data = np.asarray(fake_data)

    no, seq_len, dim = real_data.shape
    hidden_dim = max(int(dim / 2), 1)

    fake_t = torch.FloatTensor(fake_data).to(device)

    model = _Predictor(dim - 1, hidden_dim).to(device)
    optimizer = optim.Adam(model.parameters())
    criterion = nn.L1Loss()

    model.train()
    for _ in range(iterations):
        idx = torch.randperm(len(fake_t))[:batch_size]
        X_mb = fake_t[idx, :-1, :-1]   # (batch, seq_len-1, dim-1)
        Y_mb = fake_t[idx, 1:, -1:]    # (batch, seq_len-1, 1)

        optimizer.zero_grad()
        y_pred = model(X_mb)           # (batch, seq_len-1, 1)
        loss = criterion(Y_mb, y_pred)
        loss.backward()
        optimizer.step()

    model.eval()
    MAE_temp = 0.0
    with torch.no_grad():
        for i in range(no):
            x = torch.FloatTensor(real_data[i:i+1, :-1, :-1]).to(device)  # (1, seq_len-1, dim-1)
            y_true = real_data[i, 1:, -1:]                                 # (seq_len-1, 1)
            y_hat = model(x).squeeze(0).cpu().numpy()                      # (seq_len-1, 1)
            MAE_temp += mean_absolute_error(y_true, y_hat)

    return MAE_temp / no


# ---------------------------------------------------------------------------
# Composite evaluation (multiple runs for mean/std)
# ---------------------------------------------------------------------------

def evaluate_samples(real_data, fake_data, device='cuda', n_iterations=1,
                     disc_iterations=2000, pred_iterations=5000):
    """
    Compute discriminative and predictive scores.

    Args:
        real_data:       (N, seq_len, features)
        fake_data:       (N, seq_len, features)
        device:          'cuda' or 'cpu'
        n_iterations:    number of independent runs (1 matches baseline protocol)
        disc_iterations: training iterations for discriminative score
        pred_iterations: training iterations for predictive score

    Returns:
        dict with mean/std of each metric
    """
    disc_scores = []
    test_accs = []
    pred_scores = []

    print("\nComputing Discriminative Score...")
    for i in range(n_iterations):
        disc, acc = discriminative_score(real_data, fake_data,
                                         iterations=disc_iterations, device=device)
        disc_scores.append(disc)
        test_accs.append(acc)
        print(f"  Run {i+1}/{n_iterations}: disc={disc:.4f}, test_acc={acc:.4f}")

    print("\nComputing Predictive Score...")
    for i in range(n_iterations):
        pred = predictive_score(real_data, fake_data,
                                iterations=pred_iterations, device=device)
        pred_scores.append(pred)
        print(f"  Run {i+1}/{n_iterations}: MAE={pred:.4f}")

    return {
        'discriminative': {
            'scores': disc_scores,
            'mean': float(np.mean(disc_scores)),
            'std': float(np.std(disc_scores)),
            'test_acc': float(np.mean(test_accs)),
        },
        'predictive': {
            'scores': pred_scores,
            'mean': float(np.mean(pred_scores)),
            'std': float(np.std(pred_scores)),
        },
    }


# ---------------------------------------------------------------------------
# Additional metrics (unchanged)
# ---------------------------------------------------------------------------

def _kl_from_histograms(p_vals, q_vals, n_bins=50):
    """KL divergence D(P||Q) estimated from two 1-D value arrays via histograms."""
    from scipy.stats import entropy
    lo, hi = min(p_vals.min(), q_vals.min()), max(p_vals.max(), q_vals.max())
    if hi == lo:
        return 0.0
    bins = np.linspace(lo, hi, n_bins + 1)
    p_hist = np.histogram(p_vals, bins=bins)[0].astype(float) + 1e-10
    q_hist = np.histogram(q_vals, bins=bins)[0].astype(float) + 1e-10
    p_hist /= p_hist.sum()
    q_hist /= q_hist.sum()
    return float(entropy(p_hist, q_hist))


def vds_score(real_data, fake_data, n_bins=50):
    """
    Value Distribution Shift (VDS) — PaD-TS metric.
    KL divergence between per-feature value distributions, averaged across features.
    Lower is better.
    """
    F = real_data.shape[-1]
    return float(np.mean([
        _kl_from_histograms(
            real_data[:, :, i].flatten(),
            fake_data[:, :, i].flatten(),
            n_bins,
        )
        for i in range(F)
    ]))


def fdds_score(real_data, fake_data, n_bins=50):
    """
    Functional Dependency Distribution Shift (FDDS) — PaD-TS metric.
    For each feature pair, computes the per-sample Pearson cross-correlation,
    then KL divergence between real and generated distributions of those values.
    Averaged across all pairs. Lower is better.
    """
    N, L, F = real_data.shape
    pairs = [(i, j) for i in range(F) for j in range(i + 1, F)]
    if not pairs:
        return 0.0

    kl_divs = []
    for i, j in pairs:
        real_cc = np.array([np.corrcoef(real_data[n, :, i], real_data[n, :, j])[0, 1]
                            for n in range(N)])
        fake_cc = np.array([np.corrcoef(fake_data[n, :, i], fake_data[n, :, j])[0, 1]
                            for n in range(N)])
        real_cc = np.nan_to_num(real_cc, nan=0.0)
        fake_cc = np.nan_to_num(fake_cc, nan=0.0)
        kl_divs.append(_kl_from_histograms(real_cc, fake_cc, n_bins))

    return float(np.mean(kl_divs))


def correlational_score(real_data, fake_data):
    """
    Correlational score — Diffusion-TS metric.
    Absolute error between the cross-autocorrelation matrices of real and generated
    data at lag 0, summed over all feature pairs and normalised by 10.
    Lower is better.
    """
    def cacf(x):
        x = (x - x.mean(axis=(0, 1), keepdims=True)) / (x.std(axis=(0, 1), keepdims=True) + 1e-8)
        F = x.shape[2]
        r, c = np.tril_indices(F)
        return (x[:, :, r] * x[:, :, c]).mean(axis=1).mean(axis=0)

    return float(np.abs(cacf(fake_data) - cacf(real_data)).sum() / 10.0)


def compute_statistics(data, feature_names):
    """Compute basic statistics for each feature."""
    stats = {}
    for i, name in enumerate(feature_names):
        feature_data = data[:, i, :] if len(data.shape) == 3 else data[:, :, i]
        feature_data = feature_data.flatten()
        stats[name] = {
            'mean': float(np.nanmean(feature_data)),
            'std': float(np.nanstd(feature_data)),
            'min': float(np.nanmin(feature_data)),
            'max': float(np.nanmax(feature_data)),
            'median': float(np.nanmedian(feature_data)),
        }
    return stats


def calculate_statistics(data):
    """Calculate basic statistics of the data"""
    return {
        'mean': np.mean(data),
        'std': np.std(data),
        'min': np.min(data),
        'max': np.max(data),
        'median': np.median(data),
    }
