"""
Gaussian Diffusion Process — forward (noising) and reverse helpers.
Copied from MyCode and stripped to only what DecompDiff needs:
  - q_sample       : add noise to x_0 at timestep t
  - loss_weight    : per-timestep weighting (low t = high weight)
  - p_mean_variance / predict_noise_from_start : used in DDIM / DDPM samplers
"""

import torch
import torch.nn as nn
from typing import Tuple


class GaussianDiffusion(nn.Module):

    def __init__(
        self,
        num_timesteps: int = 1000,
        beta_start: float = 1e-4,
        beta_end: float = 2e-2,
        noise_schedule: str = "cosine",
        device: str = "cpu",
    ):
        super().__init__()
        self.num_timesteps = num_timesteps
        self.noise_schedule = noise_schedule
        self.device = device
        self._build_schedule(beta_start, beta_end)

    def _build_schedule(self, beta_start: float, beta_end: float):
        if self.noise_schedule == "linear":
            scale = 1000 / self.num_timesteps
            betas = torch.linspace(
                scale * beta_start, scale * beta_end,
                self.num_timesteps, dtype=torch.float64, device=self.device,
            )
        elif self.noise_schedule == "cosine":
            s = 0.008
            steps = torch.arange(self.num_timesteps + 1, device=self.device, dtype=torch.float64)
            ac = torch.cos(((steps / self.num_timesteps) + s) / (1 + s) * torch.pi * 0.5) ** 2
            ac = ac / ac[0]
            betas = torch.clip(1 - ac[1:] / ac[:-1], 0.0001, 0.9999)
        elif self.noise_schedule == "exponential":
            t = torch.arange(self.num_timesteps, device=self.device, dtype=torch.float32)
            betas = beta_start + (beta_end - beta_start) * (1 - torch.exp(-t / self.num_timesteps))
        else:
            raise ValueError(f"Unknown noise schedule: {self.noise_schedule}")

        betas = torch.clip(betas, 1e-5, 0.9999).float()
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        alphas_cumprod_prev = torch.cat([torch.ones(1, device=self.device), alphas_cumprod[:-1]])

        self.register_buffer("betas", betas)
        self.register_buffer("alphas", alphas)
        self.register_buffer("alphas_cumprod", alphas_cumprod)
        self.register_buffer("alphas_cumprod_prev", alphas_cumprod_prev)
        self.register_buffer("sqrt_alphas_cumprod",       torch.sqrt(alphas_cumprod))
        self.register_buffer("sqrt_one_minus_alphas_cumprod", torch.sqrt(1.0 - alphas_cumprod))

        posterior_variance = betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)
        self.register_buffer("posterior_variance", posterior_variance)
        self.register_buffer("posterior_log_variance_clipped",
                             torch.log(torch.clamp(posterior_variance, min=1e-20)))
        self.register_buffer("posterior_mean_coef1",
                             betas * torch.sqrt(alphas_cumprod_prev) / (1.0 - alphas_cumprod))
        self.register_buffer("posterior_mean_coef2",
                             (1.0 - alphas_cumprod_prev) * torch.sqrt(alphas) / (1.0 - alphas_cumprod))
        self.register_buffer("sqrt_recip_alphas_cumprod",  torch.sqrt(1.0 / alphas_cumprod))
        self.register_buffer("sqrt_recipm1_alphas_cumprod", torch.sqrt(1.0 / alphas_cumprod - 1))

        # Loss weight: lower t -> higher weight (matches Diffusion-TS)
        self.register_buffer(
            "loss_weight",
            torch.sqrt(alphas) * torch.sqrt(1.0 - alphas_cumprod) / betas / 100,
        )

    def _extract(self, arr: torch.Tensor, t: torch.Tensor, x_shape: Tuple) -> torch.Tensor:
        out = arr.gather(0, t.to(arr.device))
        return out.reshape(t.shape[0], *([1] * (len(x_shape) - 1)))

    def q_sample(self, x_0: torch.Tensor, t: torch.Tensor, noise: torch.Tensor = None):
        """Add noise to x_0 at timestep t.  Returns (x_t, noise)."""
        if noise is None:
            noise = torch.randn_like(x_0)
        sqrt_ac  = self._extract(self.sqrt_alphas_cumprod,           t, x_0.shape)
        sqrt_omc = self._extract(self.sqrt_one_minus_alphas_cumprod, t, x_0.shape)
        return sqrt_ac * x_0 + sqrt_omc * noise, noise

    def predict_noise_from_start(self, x_t, t, x_0):
        return (
            self._extract(self.sqrt_recip_alphas_cumprod,  t, x_t.shape) * x_t - x_0
        ) / self._extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape)

    def predict_start_from_noise(self, x_t, t, noise):
        """Inverse of predict_noise_from_start: recover x_0 from a noise prediction."""
        return (
            self._extract(self.sqrt_recip_alphas_cumprod,   t, x_t.shape) * x_t
            - self._extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape) * noise
        )

    def q_posterior_mean_variance(self, x_0, x_t, t):
        mean = (
            self._extract(self.posterior_mean_coef1, t, x_t.shape) * x_0
            + self._extract(self.posterior_mean_coef2, t, x_t.shape) * x_t
        )
        log_var = self._extract(self.posterior_log_variance_clipped, t, x_t.shape)
        return mean, log_var

    def p_mean_variance(self, x_start, x_t, t, clip_denoised=True):
        if clip_denoised:
            x_start = x_start.clamp(-1.0, 1.0)
        mean, log_var = self.q_posterior_mean_variance(x_start, x_t, t)
        return mean, log_var, x_start
