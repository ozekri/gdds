"""Log-linear noise schedule implementation."""

import torch

from .base import NoiseSchedule


class LogLinear(NoiseSchedule):
  """Log-linear noise schedule: alpha(t) = 1 - (1-eps)*t."""

  def __init__(self, eps: float = 1e-3, **kwargs):
    super().__init__()
    self.eps = float(eps)

  def alpha_t(self, t: torch.Tensor) -> torch.Tensor:
    scaled_t = (1 - self.eps) * t
    return 1 - scaled_t

  def alpha_prime_t(self, t: torch.Tensor) -> torch.Tensor:
    return -(1 - self.eps) * torch.ones_like(t)


__all__ = ['LogLinear']
