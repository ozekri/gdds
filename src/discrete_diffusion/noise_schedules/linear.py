"""Linear noise schedule implementation.

Alpha(t) decreases linearly from near 1 to near 0 with epsilon trimming.
"""

from __future__ import annotations

import torch

from .base import NoiseSchedule


class LinearNoiseSchedule(NoiseSchedule):
  """Linear attenuation schedule matching MD4 defaults.

  alpha(t) = (1 - 2*eps) * (1 - t) + eps
  alpha'(t) = - (1 - 2*eps)
  """

  def __init__(self, eps: float = 1e-4):
    super().__init__()
    self.eps = float(eps)

  def alpha_t(self, t: torch.Tensor) -> torch.Tensor:
    base = 1 - t
    return (1 - 2 * self.eps) * base + self.eps

  def alpha_prime_t(self, t: torch.Tensor) -> torch.Tensor:
    return -(1 - 2 * self.eps) * torch.ones_like(t)

  def rate_scale_factor(self, t: torch.Tensor) -> torch.Tensor:
    """Return alpha_prime_t(t) / (1 - alpha_t(t)) for FlexMDM compatibility.
    
    This is used in computing rate-based losses and sampling probabilities.
    """
    return self.alpha_prime_t(t) / (1 - self.alpha_t(t))

  def inv(self, alpha: torch.Tensor) -> torch.Tensor:
    """Return t such that alpha_t(t) = alpha (inverse function).
    
    For linear schedule: alpha = (1 - 2*eps) * (1 - t) + eps
    Solving for t: t = 1 - (alpha - eps) / (1 - 2*eps)
    """
    return 1 - (alpha - self.eps) / (1 - 2 * self.eps)

  def sample(self, shape: tuple, device: torch.device) -> torch.Tensor:
    """Sample times uniformly from [0, 1].
    
    For linear schedule, uniform sampling in t space is appropriate.
    """
    return torch.rand(shape, device=device)

  def sample_truncated(self, threshold: torch.Tensor, shape: tuple, 
                       device: torch.device) -> torch.Tensor:
    """Sample times uniformly from [threshold, 1].
    
    Args:
      threshold: Lower bound(s) for time sampling (can be batched)
      shape: Shape of samples to generate
      device: Device for tensor creation
    """
    uniform = torch.rand(shape, device=device)
    # Convert threshold to alpha space and back for proper truncation
    threshold_alpha = self.alpha_t(threshold)
    # Sample uniformly between threshold_alpha and 0 (alpha at t=1)
    sampled_alpha = uniform * (0 - threshold_alpha) + threshold_alpha
    # Convert back to time space
    return self.inv(sampled_alpha.clamp(min=self.eps))


__all__ = ["LinearNoiseSchedule"]
