"""Hybrid noise schedule implementation."""

from __future__ import annotations

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from ..forward_process.utils import sample_categorical


def _get(cfg: object, path: str, default):
  cur = cfg
  for part in path.split('.'):
    if cur is None:
      return default
    if isinstance(cur, dict):
      cur = cur.get(part, default)
    else:
      cur = getattr(cur, part, default)
  return cur if cur is not None else default


def sample_t(config, batch_size: int, eps: float | None = None, device=None):
  if eps is None:
    eps = _get(config, 'algo.t_eps', _get(config, 'model.t_eps', 1e-4))

  low_disc = bool(_get(config, 'training.low_discrepancy_sampling',
                       _get(config, 'algo.low_discrepancy_sampling', False)))
  if low_disc:
    t = torch.arange(batch_size, device=device, dtype=torch.float32) / max(batch_size, 1)
    t = (t + torch.rand(1, device=device, dtype=torch.float32)).fmod(1.0)
  else:
    t = torch.rand(batch_size, device=device, dtype=torch.float32)

  t = (1 - 2 * eps) * t + eps
  return t


class HybridDiffusion(nn.Module):

  def __init__(self, tokenizer, p_uniform: float = 0.0, clip_noise: float = 20, gamma: float = 1.0):
    super().__init__()
    self.tokenizer = tokenizer
    self.mask_id = tokenizer.mask_token_id
    self.vocab_size = int(len(tokenizer))

    p_uniform = max(math.exp(-float(clip_noise)), float(p_uniform))

    log_B = float(gamma) * math.log(2.0) + math.log(p_uniform) - math.log(1.0 - p_uniform)
    self.register_buffer('log_B', torch.tensor(float(log_B)).clip(-float(clip_noise)))
    self.register_buffer('log_gamma', torch.tensor(float(gamma)).log())

    mask = torch.zeros(self.vocab_size)
    mask[self.mask_id] = 1.0
    self.register_buffer('mask', mask, persistent=False)

    unif = (1.0 - self.mask) / max(self.vocab_size - 1, 1)
    self.register_buffer('unif', unif, persistent=False)

    pr = torch.full((self.vocab_size,), -1e3)
    pr[self.mask_id] = 0.0
    self.register_buffer('log_prior', pr - pr.logsumexp(-1, keepdim=True))

  def get_alpha_betapi(self, t: torch.Tensor, eps: float = 1e-4):
    t = t[:, None]
    t1m = 1.0 - t

    gamma = self.log_gamma.exp()
    B = self.log_B.exp()
    c_t = t.pow(gamma / 2.0) * t1m.pow(gamma / 2.0) * B
    C_t = (1.0 + c_t).clamp_min(eps)

    alpha_t = t1m / C_t
    beta_pi = (t * self.mask + c_t * self.unif) / C_t
    return alpha_t, beta_pi

  def probs_at_t(self, prs: torch.Tensor, t: torch.Tensor, eps: float = 1e-4) -> torch.Tensor:
    orig_dtype = prs.dtype
    alpha_t, beta_pi = self.get_alpha_betapi(t, eps=eps)

    probs = prs.mul(alpha_t.unsqueeze(-1))
    probs[..., :beta_pi.shape[-1]].add_(beta_pi.unsqueeze(1))
    return probs.to(orig_dtype)

  def sample_zt(self, input_ids: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
    x = F.one_hot(input_ids, num_classes=self.vocab_size).to(dtype=t.dtype)
    probs = self.probs_at_t(x, t)
    z_t = sample_categorical(probs)
    return z_t


__all__ = ['HybridDiffusion', 'sample_t']
