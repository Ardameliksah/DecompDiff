from dataclasses import dataclass
from pathlib import Path
from typing import Literal

_ROOT = Path(__file__).parent.parent          # DecompDiff/
_MYCODE = _ROOT.parent / "MyCode"             # MyCode/ for shared data utils


@dataclass
class ModelConfig:
    input_channels:  int   = 6       # OHLCAV
    sequence_length: int   = 32
    hidden_dim:      int   = 128
    num_heads:       int   = 8
    num_layers:      int   = 1       # DiT blocks per trend / season path
    num_fusion_layers: int = 1       # DiT blocks in the fusion stage (after streams meet)
    mlp_ratio:       float = 4.0
    dropout:         float = 0.0
    freq_dim:        int   = 256     # sinusoidal basis size in TimestepEmbedder


@dataclass
class DiffusionConfig:
    num_timesteps:  int   = 1000
    beta_start:     float = 1e-4
    beta_end:       float = 2e-2
    noise_schedule: Literal["linear", "cosine", "exponential"] = "cosine"


@dataclass
class TrainingConfig:
    batch_size:              int   = 64
    learning_rate:           float = 1e-4
    num_epochs:              int   = 500
    warmup_steps:            int   = 100
    weight_decay:            float = 1e-4
    gradient_clip_val:       float = 1.0
    loss_type:               Literal["l1", "mse"] = "l1"
    prediction_type:         Literal["x0", "eps"] = "x0"   # predict clean sample or noise
    use_loss_weight:         bool  = True                  # Diffusion-TS per-timestep loss weight
    lr_scheduler_type:       Literal["cosine", "linear"] = "cosine"
    validate_every_n_epochs: int   = 10
    checkpoint_dir:          str   = str(_ROOT / "output" / "checkpoints")
    log_dir:                 str   = str(_ROOT / "output" / "logs")


@dataclass
class DataConfig:
    # dataset selector — one of: stock, etth1, etth2, exchange, fmri, eeg, sine
    # (see DecompDiff/data/datasets.py). `data_path` is kept for the legacy
    # stock-CSV path; other datasets load from DecompDiff/data/ via the registry.
    dataset:        str   = "stock"
    data_root:      str   = str(_ROOT / "data")   # DecompDiff/data
    data_path:      str   = str(_MYCODE / "dataset" / "stocks_data.csv")
    train_split:    float = 1.0
    num_workers:    int   = 0
    pin_memory:     bool  = True
    neg_one_to_one: bool  = True     # MinMax to [-1, 1]
    per_window_norm: bool = False
    # sine-generation params (only used when dataset == "sine")
    sine_num:       int   = 10000
    sine_dim:       int   = 5
    sine_seed:      int   = 123


class Config:
    def __init__(self):
        self.model     = ModelConfig()
        self.diffusion = DiffusionConfig()
        self.training  = TrainingConfig()
        self.data      = DataConfig()

    def to_dict(self) -> dict:
        return {
            "model":     self.model.__dict__,
            "diffusion": self.diffusion.__dict__,
            "training":  self.training.__dict__,
            "data":      self.data.__dict__,
        }


config = Config()
