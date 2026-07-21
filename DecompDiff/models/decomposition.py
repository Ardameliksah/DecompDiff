"""
Adaptive series decomposition for DecompDiff.

Trend:    moving average, kernel = nearest-odd( round(sqrt(L)) )
Seasonal: FFT of residual, keep top-k strongest frequency bins, IFFT back
          k = round(sqrt(L))

Both are determined by the window length L, so the decomposition is
identical at any batch size — consistent between training and sampling.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class SeriesDecomposition(nn.Module):
    """
    Input:  x  (B, C, L)
    Output: trend (B, C, L),  seasonal (B, C, L)

    trend + seasonal do NOT reconstruct x exactly:
      - trend   is a smoothed version of x
      - seasonal is a band-limited version of (x - trend)
    The remainder (x - trend - seasonal) is discarded noise; the two
    DiT paths in DecompDiff learn to predict x_0 from these two signals.
    """

    def _trend_kernel(self, L: int) -> int:
        # round(sqrt(32)) = 6, but avg_pool1d requires an odd kernel to
        # preserve sequence length with symmetric padding, so we round
        # down to the nearest odd number.  For L=32: 6 -> 5.
        k = round(math.sqrt(L))
        if k % 2 == 0:
            k -= 1
        return max(k, 3)   # floor at 3 so there is always some smoothing

    def _n_freq_components(self, L: int, n_freqs: int) -> int:
        # same sqrt(L) logic as trend; clamp to available frequency bins
        return min(max(1, round(math.sqrt(L))), n_freqs)

    def forward(self, x: torch.Tensor):
        B, C, L = x.shape

        # ── Trend: moving average ─────────────────────────────────────────
        k = self._trend_kernel(L)
        # padding = k//2 gives exact output length L when k is odd
        trend = F.avg_pool1d(x, kernel_size=k, stride=1, padding=k // 2)
        # trend: (B, C, L)

        # ── Seasonal: top-k FFT components of the residual ───────────────
        residual = x - trend                                # (B, C, L)

        fft = torch.fft.rfft(residual, dim=-1)              # (B, C, L//2+1)  complex

        n_keep = self._n_freq_components(L, fft.shape[-1])
        topk_idx = fft.abs().topk(n_keep, dim=-1).indices  # (B, C, n_keep)

        mask = torch.zeros(B, C, fft.shape[-1], dtype=torch.bool, device=x.device)
        mask.scatter_(-1, topk_idx, True)

        seasonal = torch.fft.irfft(fft * mask, n=L, dim=-1)  # (B, C, L)

        return trend, seasonal


if __name__ == "__main__":
    torch.manual_seed(0)

    model = SeriesDecomposition()
    B, C, L = 32, 6, 32
    x = torch.randn(B, C, L)

    trend, seasonal = model(x)

    print(f"Input:    {x.shape}")
    print(f"Trend:    {trend.shape}")
    print(f"Seasonal: {seasonal.shape}")

    k     = model._trend_kernel(L)
    n_keep = model._n_freq_components(L, L // 2 + 1)
    print(f"\nL={L}: trend kernel={k},  freq components kept={n_keep}/{L // 2 + 1}")

    assert trend.shape == x.shape
    assert seasonal.shape == x.shape

    print("\nWindow-length scaling:")
    print(f"  {'L':>5}  {'trend_k':>8}  {'n_freqs_kept':>13}  {'total_freqs':>12}")
    for l in [16, 24, 32, 48, 64, 96, 128]:
        k_     = model._trend_kernel(l)
        n_     = model._n_freq_components(l, l // 2 + 1)
        print(f"  {l:>5}  {k_:>8}  {n_:>13}  {l // 2 + 1:>12}")

    print("\nAll checks passed.")
