"""Uniform forward process: random token replacement.

With probability `(1 - alpha_t)`, replaces each token by a uniformly drawn
token from the vocabulary (optionally excluding the mask token if present).
"""

from __future__ import annotations

import torch

from .base import ForwardProcess
from .utils import _effective_vocab_size, _mask_token_id
from ..noise_schedules.base import NoiseSchedule


class UniformForwardProcess(ForwardProcess):
  def __init__(self, tokenizer, schedule: NoiseSchedule, name=None) -> None:
    super().__init__(tokenizer=tokenizer, schedule=schedule, name=name)
    self.vocab_size = _effective_vocab_size(tokenizer)

  @torch.no_grad()
  def forward(self, input_ids: torch.Tensor, t: torch.Tensor, return_info: bool = False):
    alpha_t = self.schedule.alpha_t(t).view(-1, 1)
    p_replace = (1.0 - alpha_t).to(dtype=torch.float32)
    jump_mask = (torch.rand_like(input_ids, dtype=torch.float32) < p_replace).to(torch.bool)
    uniform_draw = torch.randint(
      low=0, high=self.vocab_size, size=input_ids.shape, device=input_ids.device, dtype=input_ids.dtype)
    xt = torch.where(jump_mask, uniform_draw, input_ids)
    if return_info:
      return xt, {"jump_mask": jump_mask}
    return xt

  def sample_prior(self, *batch_dims) -> torch.Tensor:
    size = batch_dims[0] if len(batch_dims) == 1 and isinstance(batch_dims[0], (tuple, list)) else batch_dims
    return torch.randint(low=0, high=self.vocab_size, size=tuple(size), dtype=torch.int64)

  def get_limiting_distribution(self) -> torch.Tensor:
    return torch.full((self.vocab_size,), 1.0 / float(self.vocab_size))
