"""
DecompDiff — generate .npy files for DiffusionTS evaluation.

Saves two files to output/datas/:
  decompdiff_real_<L>.npy   — real windows  (N, L, C) float32 in [0, 1]
  decompdiff_fake_<L>.npy   — generated     (N, L, C) float32 in [0, 1]

Format matches DiffusionTS exactly:
  - shape  : (N, seq_len, features)  i.e. time-first
  - range  : [0, 1]  (global MinMax scaler, same as norm_truth npy files)

Run DiffusionTS discriminative / predictive / Context-FID on these files yourself.

Usage:
    python -m DecompDiff.dts_eval --checkpoint output/checkpoints/best_model.pt
    python -m DecompDiff.dts_eval --checkpoint output/checkpoints/best_model.pt --device cuda
"""

import sys
import argparse
import numpy as np
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

from DecompDiff.models.decompDiff import DecompDiff
from DecompDiff.models.diffusion  import GaussianDiffusion
from DecompDiff.config.stocks_config import Config
from MyCode.utils.data_utils.loader import create_data_loaders


def generate_npy(
    checkpoint_path: str   = None,
    device:          str   = "cpu",
    num_steps:       int   = 50,
    eta:             float = 0.0,
    output_dir:      str   = None,
):
    cfg = Config()

    out_dir = Path(output_dir) if output_dir else Path(cfg.training.checkpoint_dir).parent / "datas"
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── model ─────────────────────────────────────────────────────────────────
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
        model.load_state_dict(ckpt["model_state_dict"])
        print(f"Loaded: {checkpoint_path}")
    else:
        print("WARNING: no checkpoint found — using random weights")

    model.eval()

    # ── real data ─────────────────────────────────────────────────────────────
    print("Loading real data...")
    train_loader, _, _ = create_data_loaders(
        csv_path       = cfg.data.data_path,
        batch_size     = 256,
        window_length  = cfg.model.sequence_length,
        neg_one_to_one = cfg.data.neg_one_to_one,
        train_ratio    = cfg.data.train_split,
        num_workers    = 0,
        per_window     = cfg.data.per_window_norm,
        pin_memory     = False,
    )
    real_CL = np.concatenate([b.cpu().numpy() for b in train_loader], axis=0)  # (N, C, L) [-1,1]
    N = real_CL.shape[0]
    print(f"  {N} windows")

    # ── generate ──────────────────────────────────────────────────────────────
    print(f"Generating {N} samples  (DDIM steps={num_steps}, eta={eta})...")
    chunk  = 256
    chunks = []
    for start in range(0, N, chunk):
        bs  = min(chunk, N - start)
        out = model.sample(diffusion, batch_size=bs, num_steps=num_steps, eta=eta).cpu()
        chunks.append(out)
    fake_CL = torch.cat(chunks, dim=0).numpy()                                 # (N, C, L) [-1,1]

    # ── convert: (N, C, L) [-1,1]  →  (N, L, C) [0,1] ──────────────────────
    # Same as DiffusionTS unnormalize_to_zero_to_one after global MinMax.
    real_np = ((real_CL.transpose(0, 2, 1) + 1.0) * 0.5).astype(np.float32)
    fake_np = ((fake_CL.transpose(0, 2, 1) + 1.0) * 0.5).astype(np.float32)

    print(f"  real [{real_np.min():.4f}, {real_np.max():.4f}]  fake [{fake_np.min():.4f}, {fake_np.max():.4f}]")

    # ── save ──────────────────────────────────────────────────────────────────
    L = cfg.model.sequence_length
    real_path = out_dir / f"decompdiff_real_{L}.npy"
    fake_path = out_dir / f"decompdiff_fake_{L}.npy"
    np.save(real_path, real_np)
    np.save(fake_path, fake_np)
    print(f"Saved: {real_path}  {real_np.shape}")
    print(f"Saved: {fake_path}  {fake_np.shape}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str,   default=None)
    parser.add_argument("--device",     type=str,   default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--steps",      type=int,   default=50)
    parser.add_argument("--eta",        type=float, default=0.0)
    parser.add_argument("--output-dir", type=str,   default=None)
    args = parser.parse_args()

    generate_npy(
        checkpoint_path = args.checkpoint,
        device          = args.device,
        num_steps       = args.steps,
        eta             = args.eta,
        output_dir      = args.output_dir,
    )
