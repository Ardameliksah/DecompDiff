"""
Dataset registry + raw loaders for DecompDiff.

Every dataset is reduced to one of two forms that the windowing pipeline
(StockDataset) can consume:

  - "series"  : a long continuous multivariate series of shape (T, F).
                StockDataset slides windows over it.
  - "windows" : already-independent windows of shape (N, L, F) (e.g. sine).
                StockDataset uses them directly (no sliding).

Feature counts are inferred from the data, not hard-coded, so the model's
input_channels can be set from whatever the dataset provides.

Supported names: stock, etth1, etth2, exchange, fmri, eeg, sine.
"""

import os
import numpy as np

# default location of the copied dataset files (DecompDiff/data)
DATA_ROOT = os.path.dirname(os.path.abspath(__file__))


# ── per-format raw loaders (return a (T, F) float32 continuous series) ─────────

def _load_stock(root):
    import pandas as pd
    df = pd.read_csv(os.path.join(root, "stocks", "stock_data.csv"))
    return df.values.astype(np.float32)                       # (3685, 6)


def _load_etth(root, which="ETTh1"):
    import pandas as pd
    df = pd.read_csv(os.path.join(root, "ETT-small", f"{which}.csv"))
    return df.values[:, 1:].astype(np.float32)                # drop date col -> (T, 7)


def _load_exchange(root):
    import pandas as pd
    df = pd.read_csv(os.path.join(root, "exchange_rate", "exchange_rate.txt"), header=None)
    return df.values.astype(np.float32)                       # (7588, 8)


def _load_fmri(root):
    from scipy.io import loadmat
    m = loadmat(os.path.join(root, "fMRI", "sim4.mat"))
    return m["ts"].astype(np.float32)                         # (10000, 50)


def _load_eeg(root):
    from scipy.io import arff
    import pandas as pd
    data, _ = arff.loadarff(os.path.join(root, "EEG", "EEG_Eye_State.arff"))
    df = pd.DataFrame(data)
    return df.values[:, :-1].astype(np.float32)               # drop label -> (14980, 14)


def generate_sine(num=10000, window=24, dim=5, seed=123):
    """Independent sine windows -> (num, window, dim) in [0, 1] (TimeGAN protocol)."""
    st0 = np.random.get_state()
    np.random.seed(seed)
    data = []
    for _ in range(num):
        cols = []
        for _k in range(dim):
            freq = np.random.uniform(0, 0.1)
            phase = np.random.uniform(0, 0.1)
            cols.append([np.sin(freq * j + phase) for j in range(window)])
        w = np.transpose(np.asarray(cols))                    # (window, dim)
        data.append((w + 1) * 0.5)                            # -> [0, 1]
    np.random.set_state(st0)
    return np.asarray(data, dtype=np.float32)


# ── registry ──────────────────────────────────────────────────────────────────
# kind: "series" -> loader returns (T, F);  "windows" -> loader returns (N, L, F)
# channels / default_window are informational (channels is verified at load time).

DATASET_INFO = {
    "stock":    {"kind": "series", "channels": 6,  "default_window": 24,
                 "loader": lambda root, **kw: _load_stock(root)},
    "etth1":    {"kind": "series", "channels": 7,  "default_window": 24,
                 "loader": lambda root, **kw: _load_etth(root, "ETTh1")},
    "etth2":    {"kind": "series", "channels": 7,  "default_window": 24,
                 "loader": lambda root, **kw: _load_etth(root, "ETTh2")},
    "exchange": {"kind": "series", "channels": 8,  "default_window": 24,
                 "loader": lambda root, **kw: _load_exchange(root)},
    "fmri":     {"kind": "series", "channels": 50, "default_window": 24,
                 "loader": lambda root, **kw: _load_fmri(root)},
    "eeg":      {"kind": "series", "channels": 14, "default_window": 24,
                 "loader": lambda root, **kw: _load_eeg(root)},
    "sine":     {"kind": "windows", "channels": 5, "default_window": 24,
                 "loader": lambda root, window=24, sine_num=10000, sine_dim=5,
                                  seed=123, **kw:
                                  generate_sine(sine_num, window, sine_dim, seed)},
}


def list_datasets():
    return sorted(DATASET_INFO.keys())


def load_dataset(name, data_root=None, window=24, sine_num=10000, sine_dim=5, seed=123):
    """
    Return (kind, data) for the named dataset:
      kind == "series"  -> data is (T, F)
      kind == "windows" -> data is (N, L, F)
    """
    name = name.lower()
    if name not in DATASET_INFO:
        raise ValueError(f"Unknown dataset '{name}'. Available: {list_datasets()}")
    info = DATASET_INFO[name]
    root = data_root or DATA_ROOT
    data = info["loader"](root, window=window, sine_num=sine_num, sine_dim=sine_dim, seed=seed)
    return info["kind"], data


def make_loaders(name, batch_size=64, window=None, train_ratio=1.0,
                 neg_one_to_one=True, per_window=False, num_workers=0,
                 pin_memory=False, data_root=None,
                 sine_num=10000, sine_dim=5, seed=123):
    """
    One-call entry point: load a named dataset and build DecompDiff DataLoaders.

    Returns (train_loader, test_loader, dataset).  Read `dataset.num_features`
    to size the model's input_channels for this dataset.
    """
    from MyCode.utils.data_utils.loader import create_data_loaders
    name = name.lower()
    window = window or DATASET_INFO[name]["default_window"]
    kind, data = load_dataset(name, data_root, window, sine_num, sine_dim, seed)
    kw = dict(batch_size=batch_size, window_length=window, neg_one_to_one=neg_one_to_one,
              train_ratio=train_ratio, per_window=per_window,
              num_workers=num_workers, pin_memory=pin_memory)
    if kind == "windows":
        return create_data_loaders(windows=data, **kw)
    return create_data_loaders(raw=data, **kw)
