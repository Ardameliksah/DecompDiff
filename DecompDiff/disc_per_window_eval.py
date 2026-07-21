"""
Per-window discriminative report for a DecompDiff checkpoint.

Chain:
    checkpoint.pt  ->  rebuild model  ->  generate fake windows
                   ->  train post-hoc GRU discriminator (TimeGAN protocol)
                   ->  classify EVERY real & fake window
                   ->  report every guess (right / wrong) + save full table

The generation / reshape / rescale steps are copied verbatim from
DecompDiff/evaluate.py so the numbers line up with the normal eval.

Usage:
    python -m DecompDiff.disc_per_window_eval \
        --checkpoint output/checkpoints/<run_name>/checkpoint_ep2000.pt \
        --device cuda --seed 0 --show wrong
"""

import sys
import argparse
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

from DecompDiff.models.decompDiff import DecompDiff
from DecompDiff.models.diffusion  import GaussianDiffusion
from DecompDiff.config.stocks_config import Config
from MyCode.utils.data_utils.loader import create_data_loaders
from MyCode.eval_metrics import discriminative_per_window


def load_model_from_checkpoint(checkpoint_path, cfg, device):
    """Rebuild the model with the ARCH SAVED IN THE CHECKPOINT, then load weights.

    Unlike evaluate.py (which builds with default arch first), we read the saved
    model config BEFORE constructing the model, so sweep runs with non-default
    hidden_dim / num_layers / num_fusion_layers load correctly.
    """
    ckpt = torch.load(checkpoint_path, map_location=device)

    saved_cfg = ckpt.get("config", {}).get("model", {})
    for k, v in saved_cfg.items():
        if hasattr(cfg.model, k):
            setattr(cfg.model, k, v)
    # window_length is stored alongside the model dict in the sweep checkpoints
    win = ckpt.get("config", {}).get("window_length", cfg.model.sequence_length)
    cfg.model.sequence_length = win

    # This script targets the residual-fusion architecture that local_sweep2
    # trains. Older checkpoints (no fusion_dit / res_input_proj) can't load.
    sd = ckpt["model_state_dict"]
    if not any("fusion_dit" in k for k in sd) or not any("res_input_proj" in k for k in sd):
        raise RuntimeError(
            f"Checkpoint '{checkpoint_path}' is from an OLDER DecompDiff "
            "architecture (no residual/fusion streams) and is incompatible with "
            "the current model. Use a residual-fusion checkpoint produced by "
            "local_sweep2.ipynb (output/checkpoints/<run_name>/checkpoint_epN.pt)."
        )

    model = DecompDiff(
        input_channels    = cfg.model.input_channels,
        sequence_length   = cfg.model.sequence_length,
        hidden_dim        = cfg.model.hidden_dim,
        num_heads         = cfg.model.num_heads,
        num_layers        = cfg.model.num_layers,
        num_fusion_layers = cfg.model.num_fusion_layers,
        mlp_ratio         = cfg.model.mlp_ratio,
        dropout           = cfg.model.dropout,
        freq_dim          = cfg.model.freq_dim,
    ).to(device)
    model.load_state_dict(sd)
    model.eval()

    # sampling convention used at training time (default x0 for older checkpoints)
    prediction_type = ckpt.get("config", {}).get("prediction_type", "x0")

    epoch = ckpt.get("epoch", "?")
    print(f"Loaded checkpoint: {checkpoint_path}")
    print(f"  epoch={epoch}  window={win}  hidden_dim={cfg.model.hidden_dim}  "
          f"NL={cfg.model.num_layers}  NF={cfg.model.num_fusion_layers}  "
          f"prediction_type={prediction_type}")
    return model, prediction_type


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--steps",  type=int,   default=50, help="DDIM sampling steps")
    parser.add_argument("--eta",    type=float, default=0.0)
    parser.add_argument("--disc-iterations", type=int, default=2000,
                        help="discriminator training iterations")
    parser.add_argument("--seed",   type=int, default=0,
                        help="fixes split + init so indices/verdicts are reproducible")
    parser.add_argument("--show", choices=["all", "wrong", "correct", "none"],
                        default="wrong", help="which per-window rows to print")
    parser.add_argument("--out", type=str, default=None,
                        help="CSV path for the full table (default: next to checkpoint)")
    args = parser.parse_args()

    if not Path(args.checkpoint).exists():
        raise FileNotFoundError(f"checkpoint not found: {args.checkpoint}")

    cfg = Config()
    device = args.device

    # ── model ──────────────────────────────────────────────────────────────
    model, prediction_type = load_model_from_checkpoint(args.checkpoint, cfg, device)
    diffusion = GaussianDiffusion(
        num_timesteps  = cfg.diffusion.num_timesteps,
        beta_start     = cfg.diffusion.beta_start,
        beta_end       = cfg.diffusion.beta_end,
        noise_schedule = cfg.diffusion.noise_schedule,
        device         = device,
    ).to(device)

    # ── real data (same loader / norm as evaluate.py) ───────────────────────
    print("\nLoading real data...")
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
    real_np = np.concatenate([b.cpu().numpy() for b in train_loader], axis=0)  # (N, C, L)
    N = real_np.shape[0]
    print(f"  real windows: {N}  shape: {real_np.shape}  (C, L)")

    # ── generate fakes (chunked, same as evaluate.py) ───────────────────────
    print(f"\nGenerating {N} fake windows  (DDIM steps={args.steps}, eta={args.eta})...")
    chunk, chunks = 256, []
    with torch.no_grad():
        for start in range(0, N, chunk):
            bs = min(chunk, N - start)
            chunks.append(model.sample(diffusion, batch_size=bs,
                                       num_steps=args.steps, eta=args.eta,
                                       prediction_type=prediction_type).cpu())
    fake_np = torch.cat(chunks, dim=0).numpy()  # (N, C, L)

    # (N, C, L) -> (N, L, C) and [-1,1] -> [0,1]
    real_m = ((real_np.transpose(0, 2, 1) + 1.0) * 0.5).astype(np.float32)
    fake_m = ((fake_np.transpose(0, 2, 1) + 1.0) * 0.5).astype(np.float32)

    # ── per-window discriminative pass ──────────────────────────────────────
    print(f"\nTraining discriminator ({args.disc_iterations} iters) and classifying "
          f"all {2 * N} windows...")
    table, disc_score, acc = discriminative_per_window(
        real_m, fake_m,
        iterations = args.disc_iterations,
        device     = device,
        seed       = args.seed,
    )

    # ── summary ─────────────────────────────────────────────────────────────
    test = table[table.split == "test"]
    print("\n" + "=" * 70)
    print("Discriminative summary")
    print("=" * 70)
    print(f"  disc_score (|acc-0.5|, test split) : {disc_score:.4f}   (target 0.0)")
    print(f"  test accuracy                      : {acc:.4f}   (target 0.5)")
    print("\n  per-window guesses (test split — the honest ones):")
    for kind in ("real", "fake"):
        sub = test[test.kind == kind]
        n_correct = int(sub.correct.sum())
        print(f"    {kind:4s}: {n_correct}/{len(sub)} caught  "
              f"({100 * n_correct / max(len(sub), 1):.1f}%)")
    fooled = test[(test.kind == "fake") & (~test.correct)]
    print(f"\n  fakes that FOOLED the discriminator (test): {len(fooled)}  "
          f"<- your most realistic generations")

    # ── save full table ─────────────────────────────────────────────────────
    out = args.out or str(Path(args.checkpoint).with_name(
        Path(args.checkpoint).stem + "_per_window.csv"))
    table.to_csv(out, index=False)
    print(f"\nFull per-window table ({len(table)} rows) saved -> {out}")

    # ── print requested rows ────────────────────────────────────────────────
    if args.show != "none":
        if args.show == "wrong":
            rows = table[~table.correct]
        elif args.show == "correct":
            rows = table[table.correct]
        else:
            rows = table
        print(f"\n{'=' * 70}\n{args.show} guesses ({len(rows)} rows)\n{'=' * 70}")
        print(f"{'kind':5s} {'idx':>5s} {'split':5s} {'prob_real':>9s} "
              f"{'pred':5s} {'correct':>7s}")
        for _, r in rows.iterrows():
            print(f"{r['kind']:5s} {int(r['index']):5d} {r['split']:5s} "
                  f"{r['prob_real']:9.3f} {r['pred']:5s} {str(bool(r['correct'])):>7s}")


if __name__ == "__main__":
    main()
