"""
DecompDiff training script.

Loss:  L1( x_0_pred, x_0 )  weighted per-timestep so that low-noise
       steps contribute more than high-noise steps (Diffusion-TS style).

Weight at timestep t:
    w(t) = sqrt(alpha_t) * sqrt(1 - alphabar_t) / beta_t / 100
    -> peaks at low t (signal mostly intact) and decays toward T.

Usage:
    python -m DecompDiff.train
    python -m DecompDiff.train --device cuda --epochs 1000
"""

import sys
import time
import argparse
import json
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from torch.utils.tensorboard import SummaryWriter

# ── allow running from repo root ──────────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).parent.parent))

from DecompDiff.models.decompDiff import DecompDiff
from DecompDiff.models.diffusion  import GaussianDiffusion
from DecompDiff.config.stocks_config import Config

# data utils reused from MyCode (unchanged, no coupling to MyCode model code)
from MyCode.utils.data_utils.loader import create_data_loaders


# ── loss ──────────────────────────────────────────────────────────────────────

def compute_loss(
    model: DecompDiff,
    diffusion: GaussianDiffusion,
    x_0: torch.Tensor,
    loss_type: str = "l1",
    prediction_type: str = "x0",
    use_loss_weight: bool = True,
) -> torch.Tensor:
    """
    Single training step loss.

    1. Sample random timesteps t.
    2. Forward diffusion: x_t = sqrt(ab_t)*x_0 + sqrt(1-ab_t)*eps
    3. Model predicts the target from (x_t, t):
         prediction_type="x0"  -> target is the clean sample x_0 (default)
         prediction_type="eps" -> target is the noise eps
    4. L1/MSE loss; if use_loss_weight, weight by w(t) so low-t steps dominate
       (Diffusion-TS style). Set False for plain unweighted loss.
    """
    B      = x_0.shape[0]
    device = x_0.device

    t = torch.randint(0, diffusion.num_timesteps, (B,), device=device)

    x_t, noise = diffusion.q_sample(x_0, t)

    model_out = model(x_t, t)
    target = x_0 if prediction_type == "x0" else noise

    if loss_type == "l1":
        loss = F.l1_loss(model_out, target, reduction="none")
    else:
        loss = F.mse_loss(model_out, target, reduction="none")

    loss = loss.mean(dim=[1, 2])            # (B,)  mean over C and L

    if use_loss_weight:
        loss = loss * diffusion.loss_weight[t]   # high at low t, low at high t

    return loss.mean()


# ── scheduler ─────────────────────────────────────────────────────────────────

def make_optimizer_and_scheduler(model, cfg, total_steps: int):
    optimizer = AdamW(
        model.parameters(),
        lr=cfg.training.learning_rate,
        weight_decay=cfg.training.weight_decay,
        betas=(0.9, 0.999),
        eps=1e-8,
    )
    warmup = LinearLR(
        optimizer,
        start_factor=1e-3,
        end_factor=1.0,
        total_iters=cfg.training.warmup_steps,
    )
    cosine = CosineAnnealingLR(
        optimizer,
        T_max=max(1, total_steps - cfg.training.warmup_steps),
        eta_min=1e-6,
    )
    scheduler = SequentialLR(
        optimizer,
        schedulers=[warmup, cosine],
        milestones=[cfg.training.warmup_steps],
    )
    return optimizer, scheduler


# ── checkpoint ────────────────────────────────────────────────────────────────

def save_checkpoint(model, optimizer, epoch, loss, path: Path, cfg_dict: dict):
    torch.save({
        "epoch":               epoch,
        "loss":                loss,
        "model_state_dict":    model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "config":              cfg_dict,
    }, path)
    print(f"  checkpoint -> {path}")


# ── training-dynamics logging ─────────────────────────────────────────────────

def log_grad_dynamics(writer, model, optimizer, step, log_hist):
    """
    Log per-layer gradient health to TensorBoard. Call AFTER loss.backward()
    and BEFORE grad clipping, so the raw (pre-clip) gradients are observed.

    Logs:
      grad_norm/<layer>     -> is the gradient reaching this layer at all?
      grad_norm/_total      -> overall signal strength
      update_ratio/<layer>  -> |lr*grad| / |weight|, ~1e-3 is healthy
      weights/<layer>, grads/<layer> (histograms, only when log_hist)
    """
    cur_lr = optimizer.param_groups[0]["lr"]
    writer.add_scalar("lr", cur_lr, step)

    total_sq = 0.0
    for name, p in model.named_parameters():
        if p.grad is None:
            continue
        g = p.grad.detach()
        gn = g.norm(2).item()
        total_sq += gn ** 2
        writer.add_scalar(f"grad_norm/{name}", gn, step)

        w_std = p.detach().std().item()
        if w_std > 0:
            writer.add_scalar(f"update_ratio/{name}", cur_lr * g.std().item() / w_std, step)

        if log_hist:
            writer.add_histogram(f"weights/{name}", p.detach(), step)
            writer.add_histogram(f"grads/{name}", g, step)

    writer.add_scalar("grad_norm/_total", total_sq ** 0.5, step)


# ── training loop ─────────────────────────────────────────────────────────────

def train(device: str = "cpu", num_epochs: int = None, batch_size: int = None,
          hidden_dim: int = None, num_layers: int = None,
          dataset: str = None, window: int = None):
    cfg = Config()

    if num_epochs  is not None: cfg.training.num_epochs  = num_epochs
    if batch_size  is not None: cfg.training.batch_size  = batch_size
    if hidden_dim  is not None: cfg.model.hidden_dim     = hidden_dim
    if num_layers  is not None: cfg.model.num_layers     = num_layers
    if dataset     is not None: cfg.data.dataset         = dataset
    if window      is not None: cfg.model.sequence_length = window

    ckpt_dir = Path(cfg.training.checkpoint_dir)
    log_dir  = Path(cfg.training.log_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True,  exist_ok=True)

    print("=" * 70)
    print("DecompDiff — training")
    print("=" * 70)
    for section, vals in cfg.to_dict().items():
        print(f"  [{section}]")
        for k, v in vals.items():
            print(f"    {k}: {v}")
    print()

    # ── data ──────────────────────────────────────────────────────────────────
    print(f"Loading data (dataset={cfg.data.dataset})...")
    from DecompDiff.data.datasets import make_loaders
    train_loader, test_loader, dataset = make_loaders(
        cfg.data.dataset,
        batch_size     = cfg.training.batch_size,
        window         = cfg.model.sequence_length,
        train_ratio    = cfg.data.train_split,
        neg_one_to_one = cfg.data.neg_one_to_one,
        per_window     = cfg.data.per_window_norm,
        num_workers    = cfg.data.num_workers,
        pin_memory     = cfg.data.pin_memory,
        data_root      = cfg.data.data_root,
        sine_num       = cfg.data.sine_num,
        sine_dim       = cfg.data.sine_dim,
        seed           = cfg.data.sine_seed,
    )
    # size the model to whatever this dataset provides
    cfg.model.input_channels = dataset.num_features
    print(f"  train batches: {len(train_loader)}  |  test batches: {len(test_loader)}"
          f"  |  channels: {cfg.model.input_channels}")

    # ── model ─────────────────────────────────────────────────────────────────
    print("Building model...")
    model = DecompDiff(
        input_channels  = cfg.model.input_channels,
        sequence_length = cfg.model.sequence_length,
        hidden_dim      = cfg.model.hidden_dim,
        num_heads       = cfg.model.num_heads,
        num_layers      = cfg.model.num_layers,
        num_fusion_layers = cfg.model.num_fusion_layers,
        mlp_ratio       = cfg.model.mlp_ratio,
        dropout         = cfg.model.dropout,
        freq_dim        = cfg.model.freq_dim,
    ).to(device)

    diffusion = GaussianDiffusion(
        num_timesteps  = cfg.diffusion.num_timesteps,
        beta_start     = cfg.diffusion.beta_start,
        beta_end       = cfg.diffusion.beta_end,
        noise_schedule = cfg.diffusion.noise_schedule,
        device         = device,
    ).to(device)

    counts = model.get_parameter_count()
    print(f"  total params: {counts['total']:,}")
    print(f"  time_embedder: {counts['time_embedder']:,}")
    print(f"  trend_path:    {counts['trend_path']:,}")
    print(f"  season_path:   {counts['season_path']:,}")
    print(f"  output_proj:   {counts['output_proj']:,}")

    # ── optimiser + scheduler ─────────────────────────────────────────────────
    total_steps = len(train_loader) * cfg.training.num_epochs
    optimizer, scheduler = make_optimizer_and_scheduler(model, cfg, total_steps)
    print(f"  total steps: {total_steps}  ({len(train_loader)} batches x {cfg.training.num_epochs} epochs)")

    # ── training-dynamics logging ─────────────────────────────────────────────
    run_name = time.strftime("%Y%m%d-%H%M%S")
    writer = SummaryWriter(log_dir=str(log_dir / "tb" / run_name))
    hist_every = 50          # log weight/grad histograms every N steps (heavier)
    global_step = 0
    # track how much the last linear layer (output_proj) moves each step
    tracked = ("output_proj", "trend_input_proj", "season_input_proj")
    prev_weights = {n: p.detach().clone()
                    for n, p in model.named_parameters() if n.startswith(tracked)}

    # ── loop ──────────────────────────────────────────────────────────────────
    history = {"train_loss": [], "val_loss": [], "lr": []}
    best_val  = float("inf")

    print("\nTraining...\n")
    for epoch in range(cfg.training.num_epochs):
        # train
        model.train()
        epoch_loss = torch.tensor(0.0, device=device)
        for batch in train_loader:
            x_0 = batch.to(device)
            optimizer.zero_grad()
            loss = compute_loss(model, diffusion, x_0, cfg.training.loss_type,
                                prediction_type=cfg.training.prediction_type,
                                use_loss_weight=cfg.training.use_loss_weight)
            loss.backward()

            # observe gradient health on the raw (pre-clip) gradients
            log_grad_dynamics(writer, model, optimizer, global_step,
                              log_hist=(global_step % hist_every == 0))

            nn.utils.clip_grad_norm_(model.parameters(), cfg.training.gradient_clip_val)
            optimizer.step()
            scheduler.step()

            # post-update: how much each tracked linear layer actually moved
            for name, p in model.named_parameters():
                if name in prev_weights:
                    delta = (p.detach() - prev_weights[name]).norm(2).item()
                    writer.add_scalar(f"weight_delta/{name}", delta, global_step)
                    prev_weights[name] = p.detach().clone()
            writer.add_scalar("loss/train_step", loss.item(), global_step)

            global_step += 1
            epoch_loss += loss.detach()

        train_loss = (epoch_loss / len(train_loader)).item()
        current_lr = optimizer.param_groups[0]["lr"]

        # validate
        val_loss = None
        if (epoch + 1) % cfg.training.validate_every_n_epochs == 0:
            model.eval()
            val_total = torch.tensor(0.0, device=device)
            with torch.no_grad():
                for batch in test_loader:
                    x_0 = batch.to(device)
                    val_total += compute_loss(model, diffusion, x_0, cfg.training.loss_type).detach()
            val_loss = (val_total / len(test_loader)).item()

            if val_loss < best_val:
                best_val = val_loss
                save_checkpoint(
                    model, optimizer, epoch + 1, val_loss,
                    ckpt_dir / "best_model.pt",
                    cfg.to_dict(),
                )

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["lr"].append(current_lr)

        if val_loss is not None:
            print(f"Epoch {epoch+1:4d} | train {train_loss:.5f} | val {val_loss:.5f} | lr {current_lr:.2e}")
        else:
            print(f"Epoch {epoch+1:4d} | train {train_loss:.5f} | lr {current_lr:.2e}")

    # ── save log ──────────────────────────────────────────────────────────────
    writer.close()
    log_path = log_dir / "training_log.json"
    with open(log_path, "w") as f:
        json.dump(history, f, indent=2)
    print(f"\nLog saved -> {log_path}")
    print(f"Best val loss: {best_val:.5f}")
    print("Done.")


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--device",  default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--epochs",  type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--hidden-dim", type=int, default=None)
    parser.add_argument("--num-layers", type=int, default=None)
    parser.add_argument("--dataset", type=str, default=None,
                        help="stock, etth1, etth2, exchange, fmri, eeg, sine")
    parser.add_argument("--window", type=int, default=None, help="window length")
    args = parser.parse_args()

    train(device=args.device, num_epochs=args.epochs, batch_size=args.batch_size,
          hidden_dim=args.hidden_dim, num_layers=args.num_layers,
          dataset=args.dataset, window=args.window)
