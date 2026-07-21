"""
DecompDiff evaluation script.

Generates samples from a trained checkpoint and computes:
  - Discriminative score  (lower = better, target 0.0)
  - Predictive MAE        (lower = better)
  - VDS                   (lower = better)
  - FDDS                  (lower = better)
  - Correlational score   (lower = better)

Metrics are identical to those used in MyCode/evaluate_unified.py so
results are directly comparable across models.

Usage:
    python -m DecompDiff.evaluate --checkpoint output/checkpoints/best_model.pt
    python -m DecompDiff.evaluate --checkpoint output/checkpoints/best_model.pt --device cuda --iterations 3
"""

import sys
import argparse
import json
import numpy as np
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

from DecompDiff.models.decompDiff import DecompDiff
from DecompDiff.models.diffusion  import GaussianDiffusion
from DecompDiff.config.stocks_config import Config

from MyCode.utils.data_utils.loader import create_data_loaders

# metrics reused directly from MyCode — same functions, same numbers
from MyCode.eval_metrics import (
    evaluate_samples,
    vds_score,
    fdds_score,
    correlational_score,
)


def evaluate(
    checkpoint_path: str = None,
    device:          str = "cpu",
    num_steps:       int = 50,
    eta:           float = 0.0,
    n_iterations:    int = 3,
    output_dir:      str = None,
):
    cfg = Config()

    out_dir = Path(output_dir or cfg.training.checkpoint_dir).parent / "eval"
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("DecompDiff — evaluation")
    print("=" * 70)

    # ── load model ────────────────────────────────────────────────────────────
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

    if checkpoint_path and Path(checkpoint_path).exists():
        ckpt = torch.load(checkpoint_path, map_location=device)
        # restore arch from checkpoint if saved
        saved_cfg = ckpt.get("config", {}).get("model", {})
        for k, v in saved_cfg.items():
            if hasattr(cfg.model, k):
                setattr(cfg.model, k, v)
        model.load_state_dict(ckpt["model_state_dict"])
        print(f"Loaded checkpoint: {checkpoint_path}")
    else:
        print("WARNING: no checkpoint — evaluating random weights")

    model.eval()
    print(f"Parameters: {model.get_parameter_count()['total']:,}")

    # ── load real data ────────────────────────────────────────────────────────
    print("\nLoading real data...")
    train_loader, _, _ = create_data_loaders(
        csv_path       = cfg.data.data_path,
        batch_size     = 256,
        window_length  = cfg.model.sequence_length,
        neg_one_to_one = cfg.data.neg_one_to_one,
        train_ratio    = cfg.data.train_split,
        num_workers    = 0,
        per_window     = cfg.data.per_window_norm,   # False → global norm
        pin_memory     = False,
    )
    real_np = np.concatenate([b.cpu().numpy() for b in train_loader], axis=0)
    N = real_np.shape[0]
    print(f"  real windows: {N}  shape: {real_np.shape}  (C, L)")

    # ── generate samples ──────────────────────────────────────────────────────
    print(f"\nGenerating {N} samples  (DDIM steps={num_steps}, eta={eta})...")
    # generate in chunks to avoid OOM
    chunk = 256
    chunks = []
    for start in range(0, N, chunk):
        bs = min(chunk, N - start)
        chunks.append(model.sample(diffusion, batch_size=bs, num_steps=num_steps, eta=eta).cpu())
    fake_np = torch.cat(chunks, dim=0).numpy()          # (N, C, L)
    print(f"  generated shape: {fake_np.shape}")

    # ── prepare for metrics: (N, C, L) → (N, L, C) and [-1,1] → [0,1] ───────
    real_m = (real_np.transpose(0, 2, 1) + 1.0) * 0.5  # (N, L, C)
    fake_m = (fake_np.transpose(0, 2, 1) + 1.0) * 0.5  # (N, L, C)

    print(f"\n  real range  [{real_m.min():.3f}, {real_m.max():.3f}]")
    print(f"  fake range  [{fake_m.min():.3f}, {fake_m.max():.3f}]")

    # ── metrics ───────────────────────────────────────────────────────────────
    results = {}

    print(f"\nComputing TimeGAN metrics ({n_iterations} runs each, N={N})...")
    metric_results = evaluate_samples(
        real_m, fake_m,
        device       = device,
        n_iterations = n_iterations,
    )
    disc = metric_results["discriminative"]
    pred = metric_results["predictive"]
    results["disc_score"]     = disc["mean"]
    results["disc_score_std"] = disc["std"]
    results["test_acc"]       = disc["test_acc"]
    results["pred_mae"]       = pred["mean"]
    results["pred_mae_std"]   = pred["std"]

    print(f"\nComputing VDS / FDDS / Correlational score...")
    results["vds"]  = vds_score(real_m, fake_m)
    results["fdds"] = fdds_score(real_m, fake_m)
    results["corr"] = correlational_score(real_m, fake_m)

    # ── print summary ─────────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("Results")
    print("=" * 70)
    print(f"  Discriminative score : {results['disc_score']:.4f} +/- {results['disc_score_std']:.4f}  (target 0.0)")
    print(f"  Test accuracy        : {results['test_acc']:.4f}  (target 0.5)")
    print(f"  Predictive MAE       : {results['pred_mae']:.4f} +/- {results['pred_mae_std']:.4f}")
    print(f"  VDS                  : {results['vds']:.4f}")
    print(f"  FDDS                 : {results['fdds']:.4f}")
    print(f"  Correlational score  : {results['corr']:.4f}")

    # ── save ──────────────────────────────────────────────────────────────────
    results_path = out_dir / "results.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved -> {results_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--device",     default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--steps",      type=int,   default=50)
    parser.add_argument("--eta",        type=float, default=0.0)
    parser.add_argument("--iterations", type=int,   default=3)
    parser.add_argument("--output-dir", type=str,   default=None)
    args = parser.parse_args()

    evaluate(
        checkpoint_path = args.checkpoint,
        device          = args.device,
        num_steps       = args.steps,
        eta             = args.eta,
        n_iterations    = args.iterations,
        output_dir      = args.output_dir,
    )
