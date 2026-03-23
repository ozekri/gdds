"""Absorbing-state forward processes."""

from __future__ import annotations

import torch
from .utils import _mask_token_id, _effective_vocab_size
from .base import ForwardProcess
from ..noise_schedules.base import NoiseSchedule


class AbsorbingForwardProcess(ForwardProcess):
  """Absorbing-state forward process.

  Replaces tokens with the mask token with probability `(1 - alpha_t)`.
  Returns the noised ids and the per-position mask probability `p_mask`.
  """

  def __init__(self, tokenizer, schedule: NoiseSchedule, name: str | None = None) -> None:
    super().__init__(tokenizer=tokenizer, schedule=schedule, name=name)
    self.mask_id = _mask_token_id(tokenizer)
    self.vocab_size = _effective_vocab_size(tokenizer)

  @torch.no_grad()
  def forward(self, input_ids: torch.Tensor, t: torch.Tensor, return_info: bool = False):
    alpha_t = self.schedule.alpha_t(t).view(-1, 1)
    p_mask = (1.0 - alpha_t).to(dtype=torch.float32)
    jump_mask = (torch.rand_like(input_ids, dtype=torch.float32) < p_mask).to(torch.bool)
    xt = torch.where(jump_mask, torch.tensor(self.mask_id, device=input_ids.device, dtype=input_ids.dtype), input_ids)
    if return_info:
      return xt, {"jump_mask": jump_mask, "p_mask": p_mask}
    return xt

  def sample_prior(self, *batch_dims) -> torch.Tensor:
    size = batch_dims[0] if len(batch_dims) == 1 and isinstance(batch_dims[0], (tuple, list)) else batch_dims
    return torch.full(tuple(size), self.mask_id, dtype=torch.int64)

  def get_limiting_distribution(self) -> torch.Tensor:
    # The true stationary distribution of an absorbing process is a delta at mask_id.
    dist = torch.zeros(self.vocab_size)
    dist[self.mask_id] = 1.0
    return dist

