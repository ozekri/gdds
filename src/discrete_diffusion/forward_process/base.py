"""Base interface for discrete forward processes.

Forward processes encapsulate tokenizer-specific details and apply a chosen
noise schedule to produce noised latent variables `z_t` (or `x_t`).

This module only defines the abstract interface; concrete implementations
will be introduced separately.
"""

from __future__ import annotations

from typing import Protocol

import torch

from ..noise_schedules.base import NoiseSchedule
from .utils import _effective_vocab_size


class ForwardProcess(torch.nn.Module):
  """Abstract base class for discrete forward noising dynamics.

  Implementations should use `self.tokenizer` and `self.schedule` to compute
  noised states for given inputs and timesteps.
  """

  def __init__(self, tokenizer, schedule: NoiseSchedule, name=None) -> None:
    super().__init__()
    self.tokenizer = tokenizer
    self.schedule = schedule
    self.name = name
    self.vocab_size = _effective_vocab_size(tokenizer)

  def forward(self, input_ids: torch.Tensor, t: torch.Tensor, return_info: bool = False):
    """Return the noised tokens at time `t`.

    Args:
        input_ids: Clean tokens [B, L].
        t: Time values [B].
        return_info: If True, returns (xt, info_dict).
        
    Returns:
        Tensor: Noised tokens xt.
        If return_info=True: (xt, info_dict) where info_dict contains 'jump_mask'.
    """
    raise NotImplementedError

  def sample_prior(self, *batch_dims) -> torch.Tensor:
    """Sample from the stationary distribution (at t=1)."""
    raise NotImplementedError

  def get_limiting_distribution(self) -> torch.Tensor:
    """Return the limiting distribution probability vector."""
    raise NotImplementedError


__all__ = ["ForwardProcess"]

