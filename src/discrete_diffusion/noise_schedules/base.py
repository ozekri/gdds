"""Base interfaces for noise schedules used in discrete diffusion.

Defines a stable `NoiseSchedule` protocol with continuous-time semantics and
an adapter `ScheduleAdapter` that preserves the call signature
`schedule(t) -> (alpha_prime_t, alpha_t)` used throughout the training code.
"""

from __future__ import annotations

from typing import Optional, Tuple

import torch


class NoiseSchedule(torch.nn.Module):
  """Abstract base class for continuous-time noise schedules.

  Implementations should return attenuation factors `alpha(t)` in (0, 1] and
  their derivative `alpha'(t)` with respect to `t`. Some schedules may also
  provide a cumulative/"total" noise measure (e.g., required by SEDD).
  """

  def __init__(self) -> None:
    super().__init__()

  def alpha_t(self, t: torch.Tensor) -> torch.Tensor:
    """Return attenuation `alpha(t)` for timesteps `t` in [0, 1].

    Args:
      t: Tensor of shape `(B,)` or broadcastable to `(B, 1)` with dtype float.

    Returns:
      Tensor matching the shape of `t` (broadcastable) with values in (0, 1].
    """
    raise NotImplementedError

  def alpha_prime_t(self, t: torch.Tensor) -> torch.Tensor:
    """Return derivative `d/dt alpha(t)` for timesteps `t`.

    Args:
      t: Tensor of shape `(B,)` or broadcastable to `(B, 1)` with dtype float.

    Returns:
      Tensor broadcastable to the shape of `t`.
    """
    raise NotImplementedError

  def total_noise(self, t: torch.Tensor) -> torch.Tensor:
    """Optional cumulative noise measure for schedules that define it.

    Implementations that do not support a total noise measure may raise
    `NotImplementedError`. This is used by SEDD-style forward processes.
    """
    raise NotImplementedError


__all__ = ["NoiseSchedule"]
