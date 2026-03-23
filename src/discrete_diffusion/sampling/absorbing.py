"""Absorbing-state sampler for MDLM."""

from __future__ import annotations

import torch

from ..forward_process.utils import sample_categorical
from .base import Sampler


class AbsorbingSampler(Sampler):
  """Sampler that mirrors Diffusion.generate_samples() for absorbing models."""

  def __init__(self, config, forward_process=None):
    self.config = config
    self.forward_process = forward_process

  def compute_posterior(self, model, x, t, dt, p_x0=None,
                        noise_removal_step=False):
    alpha_t = model.noise.alpha_t(t)
    if noise_removal_step:
      alpha_s = torch.ones_like(alpha_t)
    else:
      alpha_s = model.noise.alpha_t(t - dt)
    assert alpha_t.ndim == 2
    if p_x0 is None:
      log_p_x0 = model.forward(
        x, model._sigma_from_alphat(alpha_t))
      if self.config.sampling.use_float64:
        log_p_x0 = log_p_x0.to(torch.float64)
      p_x0 = log_p_x0.exp()

    sampled_x0 = sample_categorical(p_x0)
    prob_denoise = (alpha_s - alpha_t) / (1 - alpha_t)
    should_denoise_draw = (
      torch.rand_like(x, dtype=torch.float64, device=x.device)
      < prob_denoise)
    is_masked = (x == model.mask_id)
    should_denoise_mask = is_masked & should_denoise_draw
    _x = torch.where(should_denoise_mask, sampled_x0, x)
    out = torch.where(x != model.mask_id, x, _x)
    return p_x0, out

  @torch.no_grad()
  def generate(self, model, *, num_samples, num_steps, eps, inject_bos):
    if num_steps is None:
      num_steps = self.config.sampling.steps
    x = model.prior_sample(num_samples, model.num_tokens)
    inject_bos = self.config.sampling.inject_bos if inject_bos is None else inject_bos
    if inject_bos:
      x[:, 0] = model.tokenizer.bos_token_id

    timesteps = torch.linspace(1, eps, num_steps + 1, device=model.device)
    dt = (1 - eps) / num_steps
    p_x0_cache = None
    predictor = self.config.sampling.predictor

    for i in range(num_steps):
      t = timesteps[i] * torch.ones(
        x.shape[0], 1, device=model.device)
      if predictor == 'ddpm':
        _, x = self.compute_posterior(
          model=model, x=x, t=t, dt=dt, p_x0=None)
      elif predictor == 'ddpm_cache':
        p_x0_cache, x_next = self.compute_posterior(
          model=model, x=x, t=t, dt=dt, p_x0=p_x0_cache)
        if (not torch.allclose(x_next, x)
            or model.time_conditioning):
          p_x0_cache = None
        x = x_next
      else:
        raise ValueError(f'Unsupported predictor: {predictor}')

    t0 = timesteps[-1] * torch.ones(x.shape[0], 1, device=model.device)

    _, x = self.compute_posterior(
      model=model, x=x, t=t0, dt=None,
      p_x0=p_x0_cache,
      noise_removal_step=True)

    return x
