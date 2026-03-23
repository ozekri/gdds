"""Geometric noise schedule implementation.

Provides `total_noise(t)` and `rate_noise(t)` used by SEDD-style processes.
Alpha-based methods are intentionally not implemented since SEDD relies on
the cumulative noise parameterization.
"""

from __future__ import annotations

import torch

from .base import NoiseSchedule


class GeometricNoise(NoiseSchedule):
  """Geometric schedule parameterized by sigma bounds.

  total_noise(t) = sigma_min^(1-t) * sigma_max^t
  rate_noise(t)  = total_noise(t) * (log(sigma_max) - log(sigma_min))
  """

  def __init__(self, sigma_min: float = 1e-3, sigma_max: float = 7.0, **kwargs):
    super().__init__()
    self.register_buffer('sigma_min', torch.tensor(float(sigma_min)))
    self.register_buffer('sigma_max', torch.tensor(float(sigma_max)))

  def alpha_t(self, t: torch.Tensor) -> torch.Tensor:
    return (-self.total_noise(t)).exp()

  def alpha_prime_t(self, t: torch.Tensor) -> torch.Tensor:
    a = self.alpha_t(t)
    return -a * self.rate_noise(t)

  def total_noise(self, t: torch.Tensor) -> torch.Tensor:
    # Geometric interpolation in the sigma domain
    return (self.sigma_min ** (1 - t)) * (self.sigma_max ** t)

  def rate_noise(self, t: torch.Tensor) -> torch.Tensor:
    sig = self.total_noise(t)
    return sig * (self.sigma_max.log() - self.sigma_min.log())


__all__ = ["GeometricNoise"]
