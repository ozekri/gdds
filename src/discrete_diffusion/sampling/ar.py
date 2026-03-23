"""Autoregressive sampler for AR model."""

from __future__ import annotations

import torch

from .base import Sampler


class ARSampler(Sampler):
  """Sampler for autoregressive language models."""

  def __init__(self, config, forward_process=None):
    self.config = config

  @torch.no_grad()
  def generate(self, model, *, num_samples, num_steps, eps, inject_bos):
    """Generate samples autoregressively from left to right.
    
    Args:
      model: The AR model instance.
      num_samples: Number of samples to generate.
      num_steps: Unused for AR (kept for API compatibility).
      eps: Unused for AR (kept for API compatibility).
      inject_bos: Unused for AR (BOS is always injected).
    
    Returns:
      Generated token sequences of shape [num_samples, num_tokens].
    """
    del num_steps, eps, inject_bos  # Unused for AR
    
    # Precompute token buffer
    num_pred_tokens = model.num_tokens - 1
    x = torch.zeros(
      (num_samples, num_pred_tokens + 1),
      dtype=torch.long,
      device=model.device)
    x[:, 0] = model.tokenizer.bos_token_id
    
    # Precompute Gumbel noise for sampling
    noise = (torch.distributions.Gumbel(0, 1)
             .sample((num_samples, num_pred_tokens, model.vocab_size))
             .to(model.device))
    if self.config.sampling.use_float64:
      noise = noise.to(torch.float64)
    
    # Generate tokens autoregressively
    for i in range(num_pred_tokens):
      output = model.backbone(x[:, :i + 1], None)
      output[:, :, model.mask_id] = model.neg_infinity
      output = output.log_softmax(-1)
      y = (output[:, -1, :] + noise[:, i, :]).argmax(-1)
      x[:, i + 1] = y
    
    return x

