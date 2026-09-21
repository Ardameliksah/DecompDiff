"""
DecompDiff with a selectable third stream (the original lives in decompDiff.py).

The stream historically named "residual" carries raw x_t: nothing is subtracted, so it is an
identity / skip path, and the trend and seasonal streams duplicate part of what it supplies.
This module adds a single parameter:

    real_residual = False  ->  third stream = x_t                   (identical to decompDiff.py)
    real_residual = True   ->  third stream = x_t - trend - season  (the actual remainder)

With the flag on the three streams partition the signal exactly, trend + season + remainder == x_t.
The decomposition is computed whenever it is needed, including when use_trend and use_season are
both off, so the stream combination --R means "remainder only" instead of "raw x_t only".

Neither mode adds parameters, so state_dicts remain interchangeable with decompDiff.py.
"""

import torch
import torch.nn as nn

from .decomposition import SeriesDecomposition
from .time_embedder import TimestepEmbedder
from .dit_block import LearnablePositionalEncoding, DiTStack, FusionDiTStack


def _valid_num_heads(dim: int, requested: int) -> int:
    """Return a head count that divides `dim` (attention requires dim % heads == 0).

    If the requested count already divides dim, use it.  Otherwise fall back to
    the largest divisor of dim that is <= requested (floor at 1).  This matters
    mainly in raw mode (hidden_dim == 0) where dim = input_channels (e.g. 6) and
    the configured num_heads (e.g. 8) would not divide it.
    """
    if requested > 0 and dim % requested == 0:
        return requested
    for h in range(min(requested, dim), 0, -1):
        if dim % h == 0:
            return h
    return 1


class DecompDiffRes(nn.Module):
    """
    Decomposition-based diffusion model with a residual fusion stage.

    Forward pass (training + sampling step):
        1. Decompose x_t -> trend_t, season_t
        2. Shared TimestepEmbedder -> c
        3. Three streams, each brought to model_dim (proj + positional encoding):
             - trend  : trend_t  -> proj+PE -> DiTStack(num_layers)     -> h_trend
             - season : season_t -> proj+PE -> DiTStack(num_layers)     -> h_season
             - residual (x_t)    -> proj+PE  (no DiT blocks)            -> h_res
        4. The three streams MEET: h = h_trend + h_season + h_res
        5. Fusion: h -> DiTStack(num_fusion_layers) -> h
        6. Projection back to channels -> x_0_pred

    The trend / season stacks stay sequential and their depth is tweakable via
    `num_layers`.  The fusion stack depth is tweakable via `num_fusion_layers`.

    Hidden-dim modes:
        hidden_dim > 0  : project channels -> hidden_dim, run blocks at hidden_dim,
                          project hidden_dim -> channels at the end.  The residual
                          x_t is projected + positional-encoded the same way so it
                          matches the other two streams' dimensions.
        hidden_dim == 0 : no projection at all.  Every stage runs directly on the
                          raw `input_channels` columns (model_dim = input_channels),
                          so the residual needs no projection and there is no final
                          projection layer.

    Args:
        input_channels    : number of time series channels (C)
        sequence_length   : length of each window (L)
        hidden_dim        : model dimension; 0 => run on raw channels (no projection)
        num_heads         : attention heads (auto-adjusted to divide model_dim)
        num_layers        : DiT blocks in EACH of the trend / season paths
        num_fusion_layers : DiT blocks in the fusion stage (after the streams meet)
        mlp_ratio         : MLP hidden = model_dim * mlp_ratio
        dropout           : dropout in attention and MLP
        freq_dim          : sinusoidal basis size in TimestepEmbedder
    """

    def __init__(
        self,
        input_channels: int    = 6,
        sequence_length: int   = 32,
        hidden_dim: int        = 128,
        num_heads: int         = 8,
        num_layers: int        = 1,
        num_fusion_layers: int = 1,
        mlp_ratio: float       = 4.0,
        dropout: float         = 0.0,
        freq_dim: int          = 256,
        use_trend: bool        = True,
        use_season: bool       = True,
        use_residual: bool     = True,
        real_residual: bool    = False,
        use_aux_heads: bool    = False,
    ):
        super().__init__()

        self.input_channels  = input_channels
        self.sequence_length = sequence_length
        self.hidden_dim      = hidden_dim

        # stream ablation: which paths feed the fusion stage.
        # all off  ->  "fusion only" (raw x_t is projected straight into fusion).
        self.use_trend    = use_trend
        self.use_season   = use_season
        self.use_residual = use_residual

        # What the third stream carries.
        #   False -> raw x_t. Despite the name this is an identity / skip path:
        #            nothing is subtracted, so it duplicates the whole input.
        #   True  -> the true remainder x_t - trend - season, so the three streams
        #            partition the signal (trend + season + remainder == x_t).
        # Adds no parameters either way; state_dicts stay interchangeable.
        self.real_residual = real_residual

        # hidden_dim == 0  ->  run everything on the raw channels (no projection)
        self.use_projection = hidden_dim > 0
        model_dim = hidden_dim if self.use_projection else input_channels
        self.model_dim = model_dim

        heads = _valid_num_heads(model_dim, num_heads)

        # --- decomposition (no learnable params) ---
        self.decomp = SeriesDecomposition()

        # --- shared timestep embedder (outputs model_dim so it matches c usage) ---
        self.time_embedder = TimestepEmbedder(hidden_dim=model_dim, freq_dim=freq_dim)

        # --- input projections (Identity in raw mode) ---
        if self.use_projection:
            self.trend_input_proj  = nn.Linear(input_channels, model_dim)
            self.season_input_proj = nn.Linear(input_channels, model_dim)
            self.res_input_proj    = nn.Linear(input_channels, model_dim)
        else:
            self.trend_input_proj  = nn.Identity()
            self.season_input_proj = nn.Identity()
            self.res_input_proj    = nn.Identity()

        # --- positional encodings (applied in both modes) ---
        self.trend_pe  = LearnablePositionalEncoding(model_dim, max_len=sequence_length, dropout=dropout)
        self.season_pe = LearnablePositionalEncoding(model_dim, max_len=sequence_length, dropout=dropout)
        self.res_pe    = LearnablePositionalEncoding(model_dim, max_len=sequence_length, dropout=dropout)

        # --- trend / season DiT stacks (sequential, tweakable depth) ---
        self.trend_dit  = DiTStack(model_dim, heads, num_layers, mlp_ratio, dropout)
        self.season_dit = DiTStack(model_dim, heads, num_layers, mlp_ratio, dropout)

        # --- fusion stack (after the three streams are summed) ---
        # PaD-TS-style dense aggregation + running identity: block 0 sees the
        # summed streams x_i, block 1 sees (block-0 out + x_i), etc., and the
        # output is the sum of every fusion block's output.
        self.fusion_dit = FusionDiTStack(model_dim, heads, num_fusion_layers, mlp_ratio, dropout)

        # --- output projection back to channels (Identity in raw mode) ---
        if self.use_projection:
            self.output_proj = nn.Linear(model_dim, input_channels)
        else:
            self.output_proj = nn.Identity()

        # --- OPTIONAL auxiliary component heads (training only) ---------------
        # Off by default. When off, no modules are created, so the parameter
        # count and the state_dict are byte-identical to the original model and
        # old checkpoints load unchanged. When on, each stream's DiT output also
        # gets projected back to (B, C, L) so it can be supervised against its
        # own component of x_0. sample() never calls these.
        self.use_aux_heads = use_aux_heads
        if use_aux_heads:
            self.trend_head  = nn.Linear(model_dim, input_channels)
            self.season_head = nn.Linear(model_dim, input_channels)

    def forward(self, x_t: torch.Tensor, t: torch.Tensor, return_aux: bool = False):
        """
        Args:
            x_t        : (B, C, L)  noisy sample at diffusion timestep t
            t          : (B,)        integer diffusion timesteps
            return_aux : also return the per-stream component predictions
                         (only populated when the model was built with
                         use_aux_heads=True; otherwise an empty dict)

        Returns:
            x_0_pred            : (B, C, L)  predicted clean sample, or
            (x_0_pred, aux)     : when return_aux=True
        """
        # shared timestep condition
        c = self.time_embedder(t)                       # (B, model_dim)

        # decompose if any stream needs it - the true remainder needs it even when
        # the trend and season streams are switched off (the --R case)
        if self.use_trend or self.use_season or (self.use_residual and self.real_residual):
            trend_t, season_t = self.decomp(x_t)        # each (B, C, L)

        aux = {}
        parts = []
        if self.use_trend:                              # trend stream
            h = self.trend_input_proj(trend_t.permute(0, 2, 1))
            h = self.trend_pe(h)
            f_trend = self.trend_dit(h, c)
            if self.use_aux_heads:
                aux["trend"] = self.trend_head(f_trend).permute(0, 2, 1)
            parts.append(f_trend)
        if self.use_season:                             # season stream
            h = self.season_input_proj(season_t.permute(0, 2, 1))
            h = self.season_pe(h)
            f_season = self.season_dit(h, c)
            if self.use_aux_heads:
                aux["season"] = self.season_head(f_season).permute(0, 2, 1)
            parts.append(f_season)
        if self.use_residual:                           # third stream (proj+PE, no DiT)
            # real_residual=False -> raw x_t (identity path, original behaviour)
            # real_residual=True  -> x_t - trend - season (the actual remainder)
            res_in = (x_t - trend_t - season_t) if self.real_residual else x_t
            h = self.res_input_proj(res_in.permute(0, 2, 1))
            parts.append(self.res_pe(h))

        # the active streams meet; if none are active -> "fusion only":
        # feed the raw x_t (proj + PE) straight into the fusion stage.
        if parts:
            h = sum(parts)                              # (B, L, model_dim)
        else:
            h = self.res_pe(self.res_input_proj(x_t.permute(0, 2, 1)))

        # fusion DiT stack
        h = self.fusion_dit(h, c)                       # (B, L, model_dim)

        # project back to channels and restore (B, C, L)
        x_0_pred = self.output_proj(h)                  # (B, L, C)
        x_0_pred = x_0_pred.permute(0, 2, 1)            # (B, C, L)

        # Default return is the bare tensor, exactly as before, so sample() and
        # any existing caller are untouched. Only compute_loss opts in.
        if return_aux:
            return x_0_pred, aux
        return x_0_pred

    @torch.no_grad()
    def sample(
        self,
        diffusion,
        batch_size: int   = 16,
        num_steps:  int   = 50,
        eta:        float = 0.0,
        prediction_type: str = "x0",
    ) -> torch.Tensor:
        """
        DDIM reverse process.  eta=0 -> deterministic, eta=1 -> full stochastic.

        At every step x_t is decomposed before being fed to the model, consistent
        with how the model was trained.

        prediction_type : "x0"  -> model output IS the clean sample  x_0  (default)
                          "eps" -> model output IS the noise, so x_0 is derived from it.
        MUST match the prediction_type used during training.

        Returns:
            x_0_pred : (batch_size, C, L)
        """
        device = next(self.parameters()).device

        C = self.input_channels
        L = self.sequence_length

        x_t = torch.randn(batch_size, C, L, device=device)

        # time pairs: T-1 -> ... -> 0, same schedule as Diffusion-TS
        T     = diffusion.num_timesteps
        times = torch.linspace(-1, T - 1, steps=num_steps + 1).int().tolist()
        times = list(reversed(times))
        time_pairs = list(zip(times[:-1], times[1:]))

        for time, time_next in time_pairs:
            t = torch.full((batch_size,), time, device=device, dtype=torch.long)

            # run the model, then interpret its output per prediction_type
            model_out = self(x_t, t)                    # decomposes x_t inside forward()
            if prediction_type == "eps":
                x_start = diffusion.predict_start_from_noise(x_t, t, model_out)
            else:
                x_start = model_out
            x_start = x_start.clamp(-1.0, 1.0)

            if time_next < 0:
                x_t = x_start
                continue

            # noise prediction from the (clamped) x_0, so the clamp is respected
            pred_noise = diffusion.predict_noise_from_start(x_t, t, x_start)

            alpha      = diffusion.alphas_cumprod[time]
            alpha_next = diffusion.alphas_cumprod[time_next]

            sigma = eta * ((1 - alpha / alpha_next) * (1 - alpha_next) / (1 - alpha)).sqrt()
            c     = (1 - alpha_next - sigma ** 2).sqrt()

            noise = torch.randn_like(x_t)
            x_t   = x_start * alpha_next.sqrt() + c * pred_noise + sigma * noise

        return x_t

    def get_parameter_count(self) -> dict:
        def count(module):
            return sum(p.numel() for p in module.parameters() if p.requires_grad)

        trend_path  = count(self.trend_input_proj) + count(self.trend_pe) + count(self.trend_dit)
        season_path = count(self.season_input_proj) + count(self.season_pe) + count(self.season_dit)
        res_path    = count(self.res_input_proj) + count(self.res_pe)

        return {
            "time_embedder" : count(self.time_embedder),
            "trend_path"    : trend_path,
            "season_path"   : season_path,
            "residual_path" : res_path,
            "fusion"        : count(self.fusion_dit),
            "output_proj"   : count(self.output_proj),
            "total"         : count(self),
        }


if __name__ == "__main__":
    B, C, L = 32, 6, 32

    for hidden_dim in (128, 0):
        model = DecompDiffRes(
            input_channels    = C,
            sequence_length   = L,
            hidden_dim        = hidden_dim,
            num_heads         = 8,
            num_layers        = 1,
            num_fusion_layers = 1,
            mlp_ratio         = 4.0,
            dropout           = 0.0,
            freq_dim          = 256,
        )

        x_t = torch.randn(B, C, L)
        t   = torch.randint(0, 1000, (B,))
        x_0_pred = model(x_t, t)

        assert x_0_pred.shape == (B, C, L), f"unexpected shape {x_0_pred.shape}"

        mode = f"hidden_dim={hidden_dim}" + ("  (raw channels, no projection)" if hidden_dim == 0 else "")
        print(f"\n=== {mode}  ->  model_dim={model.model_dim} ===")
        for name, n in model.get_parameter_count().items():
            print(f"  {name:<16}: {n:>10,}")
