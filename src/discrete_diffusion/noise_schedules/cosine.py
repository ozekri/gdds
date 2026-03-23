"""Cosine noise schedule implementation."""

from __future__ import annotations

import torch

from .base import NoiseSchedule


class CosineNoiseSchedule(NoiseSchedule):
  """Cosine-shaped retention schedule with epsilon trimming.

  alpha_base(t) = 1 - cos(pi/2 * (1 - t))
  alpha(t) = (1 - 2*eps) * alpha_base(t) + eps
  alpha'(t) = (1 - 2*eps) * d/dt[alpha_base(t)]
            = (1 - 2*eps) * ( - (pi/2) * sin(pi/2 * (1 - t)) )
  """

  def __init__(self, eps: float = 1e-4):
    super().__init__()
    self.eps = float(eps)

  def alpha_t(self, t: torch.Tensor) -> torch.Tensor:
    base = 1 - torch.cos(torch.pi / 2 * (1 - t))
    return (1 - 2 * self.eps) * base + self.eps

  def alpha_prime_t(self, t: torch.Tensor) -> torch.Tensor:
    base_prime = -(torch.pi / 2) * torch.sin(torch.pi / 2 * (1 - t))
    return (1 - 2 * self.eps) * base_prime


__all__ = ["CosineNoiseSchedule"]

