import math
import torch
import torch.nn as nn


class SinusoidalEmbedding(nn.Module):
    """Fixed sinusoidal basis — no learnable parameters."""

    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        device = t.device
        half = self.dim // 2
        freqs = torch.exp(
            -math.log(10000) * torch.arange(half, device=device) / (half - 1)
        )
        args = t.float().unsqueeze(1) * freqs.unsqueeze(0)
        return torch.cat([torch.sin(args), torch.cos(args)], dim=-1)


class TimestepEmbedder(nn.Module):
    """
    t (B,)  →  c (B, hidden_dim)

    Flow:
        SinusoidalEmbedding(freq_dim)      [no params]
        Linear(freq_dim  → hidden_dim)
        SiLU
        Linear(hidden_dim → hidden_dim)

    Args:
        hidden_dim : output size — must match the model hidden dimension
        freq_dim   : sinusoidal basis size (adjustable; default 256)
                     set equal to hidden_dim if you want no expansion at all
    """

    def __init__(self, hidden_dim: int, freq_dim: int = 256):
        super().__init__()
        self.sinusoidal = SinusoidalEmbedding(freq_dim)
        self.mlp = nn.Sequential(
            nn.Linear(freq_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        return self.mlp(self.sinusoidal(t))


if __name__ == "__main__":
    for hidden_dim, freq_dim in [(128, 256), (128, 128), (256, 256)]:
        emb = TimestepEmbedder(hidden_dim=hidden_dim, freq_dim=freq_dim)
        t   = torch.randint(0, 1000, (32,))
        c   = emb(t)
        params = sum(p.numel() for p in emb.parameters())
        print(f"hidden={hidden_dim}, freq={freq_dim} -> c:{c.shape}  params:{params:,}")
