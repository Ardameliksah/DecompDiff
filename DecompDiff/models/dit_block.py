import torch
import torch.nn as nn


class LearnablePositionalEncoding(nn.Module):
    """
    Learnable PE applied once before the first DiT block.
    Not used inside DiTBlock itself.

    Args:
        hidden_dim : model dimension
        max_len    : maximum sequence length
        dropout    : applied after adding PE
    """

    def __init__(self, hidden_dim: int, max_len: int = 1024, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.pe = nn.Parameter(torch.empty(1, max_len, hidden_dim))
        nn.init.uniform_(self.pe, -0.02, 0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, L, hidden_dim)
        x = x + self.pe[:, : x.size(1), :]
        return self.dropout(x)


class DiTBlock(nn.Module):
    """
    Single adaLN-Zero DiT block.

    All 6 modulation parameters (shift_msa, scale_msa, gate_msa,
    shift_mlp, scale_mlp, gate_mlp) are produced from the timestep
    condition c via one shared linear.  The linear is zero-initialised
    so every gate starts at 0 — the block is an identity at step 0.

    Args:
        hidden_dim : model dimension
        num_heads  : attention heads (hidden_dim % num_heads == 0)
        mlp_ratio  : MLP hidden = hidden_dim * mlp_ratio
        dropout    : dropout in attention and MLP

    Inputs:
        x : (B, L, hidden_dim)   sequence
        c : (B, hidden_dim)      timestep condition from TimestepEmbedder
    Output:
        x : (B, L, hidden_dim)
    """

    def __init__(
        self,
        hidden_dim: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
    ):
        super().__init__()

        self.norm1 = nn.LayerNorm(hidden_dim, elementwise_affine=False, eps=1e-6)
        self.norm2 = nn.LayerNorm(hidden_dim, elementwise_affine=False, eps=1e-6)

        self.attn = nn.MultiheadAttention(
            hidden_dim, num_heads, dropout=dropout, batch_first=True
        )

        mlp_hidden = int(hidden_dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, mlp_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden, hidden_dim),
            nn.Dropout(dropout),
        )

        # Produces all 6 modulation params from c in one shot.
        # Zero-init: gates = 0 at init → block output = 0 → pure residual path.
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_dim, 6 * hidden_dim, bias=True),
        )
        nn.init.zeros_(self.adaLN_modulation[1].weight)
        nn.init.zeros_(self.adaLN_modulation[1].bias)

    def forward(self, x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (
            self.adaLN_modulation(c).chunk(6, dim=-1)
        )

        # unsqueeze to (B, 1, hidden_dim) for broadcast over L
        scale_msa = scale_msa.unsqueeze(1)
        shift_msa = shift_msa.unsqueeze(1)
        gate_msa  = gate_msa.unsqueeze(1)
        scale_mlp = scale_mlp.unsqueeze(1)
        shift_mlp = shift_mlp.unsqueeze(1)
        gate_mlp  = gate_mlp.unsqueeze(1)

        # --- attention branch ---
        h = self.norm1(x) * (1 + scale_msa) + shift_msa
        attn_out, _ = self.attn(h, h, h)
        x = x + gate_msa * attn_out

        # --- MLP branch ---
        h = self.norm2(x) * (1 + scale_mlp) + shift_mlp
        x = x + gate_mlp * self.mlp(h)

        return x


class DiTStack(nn.Module):
    """
    N DiT blocks in sequence, all sharing the same architecture.
    The same timestep condition c is passed into every block.

    Args:
        hidden_dim : model dimension
        num_heads  : attention heads
        num_layers : number of DiT blocks
        mlp_ratio  : MLP expansion ratio
        dropout    : dropout rate
    """

    def __init__(
        self,
        hidden_dim: int,
        num_heads: int,
        num_layers: int,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.blocks = nn.ModuleList([
            DiTBlock(hidden_dim, num_heads, mlp_ratio, dropout)
            for _ in range(num_layers)
        ])

    def forward(self, x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        for block in self.blocks:
            x = block(x, c)
        return x


class FusionDiTStack(nn.Module):
    """
    Fusion-only DiT stack with PaD-TS-style dense aggregation + running identity.

    Meant for the point where the trend / season / residual streams meet.  The
    input is  x_i = h_trend + h_season + h_res.  Instead of a plain sequential
    stack, this reproduces PaD-TS's Decoder flow:

        identity = x_i
        out      = 0
        for block in blocks:
            x   = block(x, c)      # this block's output
            out = out + x          # accumulate every block's output (dense readout)
            x   = x + identity     # re-inject the running identity
            identity = x
        return out

    Concretely, block 0 sees x_i, block 1 sees (block-0 output + x_i), block 2
    sees (block-1 output + running identity), and so on; the return value is the
    SUM of all block outputs (not just the last).

    With num_layers == 1 this reduces exactly to a single DiT block: out = B0(x_i).

    Args:
        hidden_dim : model dimension
        num_heads  : attention heads
        num_layers : number of fusion DiT blocks
        mlp_ratio  : MLP expansion ratio
        dropout    : dropout rate
    """

    def __init__(
        self,
        hidden_dim: int,
        num_heads: int,
        num_layers: int,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.blocks = nn.ModuleList([
            DiTBlock(hidden_dim, num_heads, mlp_ratio, dropout)
            for _ in range(num_layers)
        ])

    def forward(self, x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        identity = x
        out = torch.zeros_like(x)
        for block in self.blocks:
            x = block(x, c)          # output of this block
            out = out + x            # dense aggregation of every block's output
            x = x + identity         # running-identity residual for the next block
            identity = x
        return out


if __name__ == "__main__":
    import sys, os
    sys.path.insert(0, os.path.dirname(__file__))
    from time_embedder import TimestepEmbedder

    B, L, C = 32, 32, 6
    hidden_dim = 128
    num_heads  = 8
    num_layers = 4

    embedder = TimestepEmbedder(hidden_dim=hidden_dim, freq_dim=256)
    pe        = LearnablePositionalEncoding(hidden_dim=hidden_dim, max_len=L)
    stack     = DiTStack(hidden_dim=hidden_dim, num_heads=num_heads, num_layers=num_layers)

    t = torch.randint(0, 1000, (B,))
    x = torch.randn(B, L, hidden_dim)

    c   = embedder(t)       # (B, hidden_dim)
    x   = pe(x)             # (B, L, hidden_dim)
    out = stack(x, c)       # (B, L, hidden_dim)

    assert out.shape == (B, L, hidden_dim), f"unexpected shape {out.shape}"

    emb_p   = sum(p.numel() for p in embedder.parameters())
    pe_p    = sum(p.numel() for p in pe.parameters())
    block_p = sum(p.numel() for p in stack.blocks[0].parameters())
    stack_p = sum(p.numel() for p in stack.parameters())

    print(f"hidden_dim={hidden_dim}, heads={num_heads}, layers={num_layers}")
    print(f"  TimestepEmbedder     : {emb_p:>10,}")
    print(f"  LearnablePE          : {pe_p:>10,}")
    print(f"  DiTBlock (×1)        : {block_p:>10,}")
    print(f"  DiTStack (×{num_layers})       : {stack_p:>10,}")
    print(f"  output shape         : {out.shape}")
    print("Test passed!")
