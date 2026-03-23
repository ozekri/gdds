import hydra.utils
import torch
import torch.nn.functional as F

from . import base as trainer_base
from ..noise_schedules import LogLinear
from ..forward_process.uniform import UniformForwardProcess


class UDLM(trainer_base.Diffusion):
  """Uniform Discrete Latent Model (UDLM).

  - Forward process: with prob (1 - alpha_t), replace token with a uniform token
    over the vocabulary; otherwise keep it unchanged.
  - Parameterization: reuse 'subs' head; logits are turned into log-probabilities
    with log_softmax and no special mask handling.
  """

  def __init__(self, config, tokenizer):
    super().__init__(config, tokenizer)
    # Limiting distribution π(y) = 1 / V for all tokens y
    self.register_buffer(
      'limiting_distribution',
      torch.full((self.vocab_size,), 1.0 / float(self.vocab_size)))
    # Config for whether to include reconstruction loss (default: False for UDLM)
    self.zero_recon_loss = getattr(config.algo, 'zero_recon_loss', True)
    self._validate_configuration()
    # Build a forward process using Hydra instantiation. If the config doesn't specify
    # one, default UDLM to the 'uniform' forward process to match semantics.
    try:
      fp = hydra.utils.instantiate(
        self.config.algo.forward_process,
        tokenizer=self.tokenizer,
        schedule=self.noise
      )
    except Exception:
      fp = None
    self._forward_process = (
      fp if isinstance(fp, UniformForwardProcess)
      else UniformForwardProcess(tokenizer=self.tokenizer, schedule=self.noise, name='uniform')
    )

  def _validate_configuration(self):
    super()._validate_configuration()
    # UDLM uses no time-conditioning by default and subs parameterization
    if self.time_conditioning:
      raise ValueError('UDLM expects algo.time_conditioning=False')
    # Only log-linear noise is supported for UDLM currently
    # This constraint exists because UDLM's loss computation hardcodes the log-linear
    # schedule form (see nll_per_token method). Other schedules may be supported in the future.
    if not isinstance(self.noise, LogLinear):
      raise ValueError(
        'UDLM currently supports only LogLinear noise schedule. '
        'Set config.algo.noise_schedule.name=log_linear')

  def prior_sample(self, *batch_dims):
    # Uniform prior over [0, vocab_size)
    return torch.randint(
      low=0,
      high=self.vocab_size,
      size=batch_dims,
      device=self.device,
      dtype=torch.int64,
    )

  def q_xt(self, x, t, sampling_eps_min=None, sampling_eps_max=None):
    del sampling_eps_min, sampling_eps_max
    # Route through the forward-process registry (uniform replacement).
    out = self._forward_process(x, t)
    xt = out[0] if isinstance(out, (tuple, list)) else out
    if getattr(self, 'ignore_bos', False):
      xt[:, 0] = x[:, 0]
    return xt

  def _process_model_output(self, model_output, xt, sigma):
    # No mask handling; UDLM uses plain log-probabilities over vocab
    return torch.log_softmax(model_output, dim=-1)

  def nll_per_token(self, log_x_theta, xt, x0, alpha_t, dalpha_t, low_var=False):
    del low_var, dalpha_t
    # Shapes
    #  log_x_theta: (B, L, V)
    #  xt, x0: (B, L)
    #  alpha_t: (B, 1) from our LogLinear schedule: alpha_t = 1 - (1 - eps) * t
    B, L, V = log_x_theta.shape
    vocab_size = V

    # Hardcode loglinear continuous-time forms (match guidance repo):
    #   alpha_t = 1 - t
    #   alpha_t' = -1
    # Recover t from our alpha_t definition: alpha_t = 1 - (1 - eps) * t
    eps = getattr(self.noise, 'eps', 1e-3)
    t = (1 - alpha_t.to(log_x_theta.dtype)) / (1 - eps)
    alpha_t_prime = -1.
    alpha_t = 1. - t[..., None]  # B, 1, 1
    x_bar = vocab_size * alpha_t * F.one_hot(x0, self.vocab_size).float() + 1 - alpha_t
    x_bar_theta = vocab_size * alpha_t * log_x_theta.exp() + 1 - alpha_t

    # α_t' / (N*α_t) with α_t' = -1
    coeff = alpha_t_prime / (vocab_size * alpha_t)

    # Term 1: indices where z_t = 1
    x_bar_zt = torch.gather(x_bar, -1, xt[..., None])  # (B, L, 1)
    x_bar_theta_zt = torch.gather(x_bar_theta, -1, xt[..., None])  # (B, L, 1)
    term1 = (vocab_size / x_bar_zt) - (vocab_size / x_bar_theta_zt)  # (B, L, 1)

    # Term 2: indices where z_t = 0
    term2 = (
      (x_bar / x_bar_zt) * (
        x_bar_theta_zt.log() - x_bar_theta.log() +
        x_bar.log() - x_bar_zt.log()
      )
    ).sum(dim=-1, keepdim=True)  # (B, L, 1)

    diffusion_loss = (coeff * (term1 - term2)).squeeze(-1)  # (B, L)

    # Optionally include reconstruction term at t=0 based on config
    if self.zero_recon_loss:
      # For UDLM with log-linear schedule, we only return the diffusion loss.
      # This is the correct formulation for continuous-time UDLM (equivalent to
      # zero_recon_loss=True in the discrete-diffusion-guidance implementation).
      return diffusion_loss
    else:
      # Include reconstruction loss (used for discrete-time or other variants)
      reconstruction_loss = self._reconstruction_loss(x0)  # (B, L)
      return diffusion_loss + reconstruction_loss

  def _metric_denoising_mask(self, xt, x0, jump_mask, valid_tokens):
    # For UDLM, denoising PPL is defined to match full validation PPL.
    del xt, x0, jump_mask
    return valid_tokens.bool()
