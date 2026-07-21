"""
Stock time series dataset.

Responsibilities (single-responsibility):
  - Load CSV once
  - Normalize via an injected BaseScaler
  - Create sliding windows
  - Temporal train/test split

The class does NOT decide which scaler to use — that is the caller's job
(see `loader.py` or `__init__.py` helpers).
"""

import numpy as np
import torch
from pathlib import Path
from torch.utils.data import Dataset

from .scalers import BaseScaler


class StockDataset(Dataset):
    """
    PyTorch Dataset for windowed stock time series.

    Data layout returned by __getitem__:
        tensor of shape (num_features, window_length)   — channels-first.
    """

    def __init__(
        self,
        csv_path: str = None,
        scaler: BaseScaler = None,
        window_length: int = 32,
        stride: int = 1,
        train_ratio: float = 0.8,
        per_window: bool = False,
        raw: np.ndarray = None,
        windows: np.ndarray = None,
    ):
        """
        Three input modes (checked in order):
          - windows : pre-made windows of shape (N, L, F) — used directly (e.g. sine).
          - raw     : a continuous series of shape (T, F) — sliding windows are built.
          - csv_path: path to a CSV — loaded then treated like `raw`.
        Output windows are channels-first (N, F, W), normalized to [-1, 1].
        """
        super().__init__()
        self.scaler = scaler
        self.window_length = window_length

        if windows is not None:
            # ── pre-windowed data (N, L, F): normalize + go straight to split ──
            windows = np.asarray(windows, dtype=np.float32)
            N, L, F = windows.shape
            self.num_features = F
            self.window_length = L
            if per_window:
                cf, w_mins, w_maxs = [], [], []
                for w in windows:                       # (L, F)
                    w_min, w_max = w.min(axis=0), w.max(axis=0)
                    w_range = w_max - w_min
                    w_range[w_range == 0] = 1.0
                    cf.append((2.0 * (w - w_min) / w_range - 1.0).T)   # (F, L)
                    w_mins.append(w_min); w_maxs.append(w_max)
                self.window_min = np.array(w_mins, dtype=np.float32)
                self.window_max = np.array(w_maxs, dtype=np.float32)
                windows = np.array(cf, dtype=np.float32)
            else:
                flat = scaler.fit_transform(windows.reshape(-1, F))    # (N*L, F)
                windows = flat.reshape(N, L, F).transpose(0, 2, 1).astype(np.float32)
        else:
            # ── continuous series: from `raw` array or CSV ──────────────────
            if raw is None:
                import pandas as pd
                csv_path = Path(csv_path)
                if not csv_path.exists():
                    raise FileNotFoundError(f"CSV not found: {csv_path}")
                df = None
                for enc in ("utf-8", "latin-1", "iso-8859-1"):
                    try:
                        df = pd.read_csv(csv_path, encoding=enc,
                                         on_bad_lines="skip", engine="python")
                        break
                    except Exception:
                        df = None
                if df is None:
                    raise ValueError(f"Could not read CSV: {csv_path}")
                raw = df.values.astype(np.float32)
            raw = np.asarray(raw, dtype=np.float32)
            self.num_features = raw.shape[1]

            # --- normalization + sliding windows -----------------------------
            if per_window:
                # Per-window MinMax [-1, 1]: each window normalized by its own stats.
                windows, w_mins, w_maxs = [], [], []
                for i in range(0, len(raw) - window_length + 1, stride):
                    w = raw[i: i + window_length]           # (W, F)
                    w_min = w.min(axis=0)                   # (F,)
                    w_max = w.max(axis=0)                   # (F,)
                    w_range = w_max - w_min
                    w_range[w_range == 0] = 1.0             # constant feature → map to 0
                    w_norm = 2.0 * (w - w_min) / w_range - 1.0   # [-1, 1]
                    windows.append(w_norm.T)                # (F, W) channels-first
                    w_mins.append(w_min)
                    w_maxs.append(w_max)
                self.window_min = np.array(w_mins, dtype=np.float32)  # (N, F)
                self.window_max = np.array(w_maxs, dtype=np.float32)  # (N, F)
            else:
                # Global scaler: fit on entire series, then slide windows
                normalized = scaler.fit_transform(raw)      # (T, F)
                windows = []
                for i in range(0, len(normalized) - window_length + 1, stride):
                    w = normalized[i: i + window_length]    # (W, F)
                    windows.append(w.T)                     # (F, W) channels-first

            windows = np.array(windows, dtype=np.float32)

        # --- temporal split ---------------------------------------------------
        # train_ratio=1.0 (DiffusionTS protocol): all data used for training;
        # test_windows mirrors train_windows so evaluation also uses all data.
        split = int(len(windows) * min(train_ratio, 1.0))
        self.train_windows = torch.from_numpy(windows[:split])
        if train_ratio >= 1.0:
            self.test_windows = self.train_windows   # no held-out set
            print(f"StockDataset: {len(windows)} windows  "
                  f"(train={len(self.train_windows)}, test=all [train_ratio=1.0])")
        else:
            self.test_windows = torch.from_numpy(windows[split:])
            print(f"StockDataset: {len(windows)} windows  "
                  f"(train={len(self.train_windows)}, test={len(self.test_windows)})")

    # ------ public helpers ---------------------------------------------------

    def get_train_data(self) -> torch.Tensor:
        return self.train_windows

    def get_test_data(self) -> torch.Tensor:
        return self.test_windows

    def denormalize(self, data, window_indices=None) -> np.ndarray:
        """
        Reverse-transform an array of shape (B, F, W) back to original scale.

        Args:
            data: (B, F, W) normalized tensor or array
            window_indices: list of int of length B — required when per_window=True
                            to select the correct per-window scale parameters.
        """
        if not isinstance(data, np.ndarray):
            data = data.cpu().numpy()
        B, F, W = data.shape

        if hasattr(self, 'window_min') and window_indices is not None:
            flat = data.transpose(0, 2, 1)              # (B, W, F)
            out = np.zeros_like(flat)
            for b, idx in enumerate(window_indices):
                w_range = self.window_max[idx] - self.window_min[idx]  # (F,)
                w_range[w_range == 0] = 1.0
                out[b] = (flat[b] + 1.0) / 2.0 * w_range + self.window_min[idx]
            return out.transpose(0, 2, 1).astype(np.float32)   # (B, F, W)

        # Fallback: global scaler
        flat = data.transpose(0, 2, 1).reshape(-1, F)       # (B*W, F)
        inv = self.scaler.inverse_transform(flat)            # (B*W, F)
        return inv.reshape(B, W, F).transpose(0, 2, 1).astype(np.float32)

    # ------ Dataset interface (training windows by default) ------------------

    def __len__(self) -> int:
        return len(self.train_windows)

    def __getitem__(self, idx: int) -> torch.Tensor:
        return self.train_windows[idx]
