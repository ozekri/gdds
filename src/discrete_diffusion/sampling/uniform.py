"""Uniform (UDLM) sampler abstraction."""

from __future__ import annotations

import torch
import torch.nn.functional as F

from ..forward_process.utils import sample_categorical
from .base import Sampler


class UniformSampler(Sampler):
  """Sampler whose updates mirror UDLM's uniform transition."""

  def __init__(self, config, forward_process=None):
    self.config = config

  def compute_posterior(self, model, x, t, dt, p_x0=None,
                        noise_removal_step=False):
    alpha_t = model.noise.alpha_t(t)
    if noise_removal_step:
      alpha_s = torch.ones_like(alpha_t)
    else:
      alpha_s = model.noise.alpha_t(t - dt)
    if p_x0 is None:
      sigma_t = model._sigma_from_alphat(alpha_t)
      log_p_x0 = model.forward(xt=x, sigma=sigma_t)
      if self.config.sampling.use_float64:
        log_p_x0 = log_p_x0.to(torch.float64)
      p_x0 = log_p_x0.exp()

    V = model.vocab_size
    alpha_t3 = alpha_t[..., None]
    alpha_s3 = alpha_s[..., None]
    alpha_ts = alpha_t3 / alpha_s3
    xt_one_hot = F.one_hot(x, V)
    limiting = model.limiting_distribution.view(1, 1, -1)

    numerator = (
      (alpha_t3 * V * p_x0 * xt_one_hot)
      + ((alpha_ts - alpha_t3) * xt_one_hot)
      + ((alpha_s3 - alpha_t3) * p_x0)
      + ((1 - alpha_ts) * (1 - alpha_s3) * limiting)
    )
    denom = (
      (alpha_t3 * V * torch.gather(p_x0, -1, x[..., None]))
      + (1 - alpha_t3)
    )
    q_xs = numerator / denom

    xs = sample_categorical(q_xs)
    return p_x0, xs

  @torch.no_grad()
  def generate(self, model, *, num_samples, num_steps, eps, inject_bos):
    if num_steps is None:
      num_steps = self.config.sampling.steps
    x = model.prior_sample(num_samples, model.num_tokens)
    if inject_bos is None:
      inject_bos = self.config.sampling.inject_bos
    if inject_bos:
      x[:, 0] = model.tokenizer.bos_token_id

    timesteps = torch.linspace(
      1, eps, num_steps + 1, device=model.device)
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
        raise ValueError(
          f'Uniform sampler only supports ddpm predictors, got {predictor}')

    t0 = timesteps[-1] * torch.ones(x.shape[0], 1,
                                    device=model.device)
    _, x = self.compute_posterior(
      model=model, x=x, t=t0, dt=None, p_x0=p_x0_cache,
      noise_removal_step=True)
    return x
