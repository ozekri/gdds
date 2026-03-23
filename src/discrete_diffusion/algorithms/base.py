import itertools
import os
import inspect
from dataclasses import dataclass
from pathlib import Path

import hydra.utils
import lightning as L
import numpy as np
import torch
import torch.nn.functional as F
import transformers

from ..evaluations import Metrics
from ..models import create_ema
from .. import utils
import omegaconf
from ..forward_process.utils import _effective_vocab_size, _unsqueeze


def ensure_mask_token(tokenizer):
  """Return mask token id and vocab size, ensuring the tokenizer exposes the mask."""
  vocab_size = _effective_vocab_size(tokenizer)
  if getattr(tokenizer, 'mask_token', None) is None:
    mask_id = vocab_size
    vocab_size += 1
  else:
    mask_id = tokenizer.mask_token_id
  if getattr(tokenizer, 'mask_token_id', None) is None:
    setattr(tokenizer, 'mask_token_id', int(mask_id))
  return int(mask_id), vocab_size

@dataclass
class Loss:
  loss: torch.FloatTensor
  nlls: torch.FloatTensor
  num_tokens: torch.FloatTensor


class TrainerBase(L.LightningModule):
  """Base Trainer class for discrete diffusion models.
  
  Handles initialization of backbone, noise schedule, and sampler, as well as
  Lightning hooks for training and validation loops.
  """
  def __init__(self, config, tokenizer: transformers.PreTrainedTokenizer, vocab_size=None):
    super().__init__()
    self.save_hyperparameters()
    self.config = config
    self.ignore_bos = getattr(self.config.algo, 'ignore_bos', False)
    self.loss_type = getattr(self.config.algo, 'loss_type', None)
    self.tokenizer = tokenizer
    if vocab_size is None:
      self.vocab_size = len(self.tokenizer)
    else:
      self.vocab_size = vocab_size
    self.sampler = self.config.sampling.predictor
    self.antithetic_sampling = self.config.training.antithetic_sampling
    self.parameterization = self.config.algo.parameterization
    self._sampler_cfg = self._resolve_sampler_config()
    self._sampler = None
    
    target = self.config.model._target_
    instantiate_config = omegaconf.OmegaConf.create({'_target_': target})
    self.backbone = hydra.utils.instantiate(
      instantiate_config,
      self.config,
      self.vocab_size,
      _recursive_=False
    )
    self.model = self.backbone

    self.T = self.config.algo.T
    self.num_tokens = self.config.model.length
    self.softplus = torch.nn.Softplus()
    self.p_nucleus = self.config.sampling.p_nucleus
    # Noise schedule - use Hydra instantiation
    # HybridDiffusion needs tokenizer passed at runtime
    if hasattr(self.config.noise, '_target_') and 'HybridDiffusion' in self.config.noise._target_:
      self.noise = hydra.utils.instantiate(self.config.noise, tokenizer=self.tokenizer)
    else:
      self.noise = hydra.utils.instantiate(self.config.noise)

    self.metrics = Metrics()

    self._prepare_ema()
    self.lr = self.config.optim.lr
    self.sampling_eps = self.config.training.sampling_eps
    self.time_conditioning = self.config.algo.time_conditioning
    if config.neg_infinity_mode == 'large-finite':
      self.neg_infinity = -1000000.0
    elif config.neg_infinity_mode == 'true-inf':
      self.neg_infinity = -float('inf')
    else:
      raise ValueError(f"neg_infinity_mode must be 'large-finite' or 'true-inf', got '{config.neg_infinity_mode}'")
    
    self._val_cache = None
    self.fast_forward_epochs = None
    self.fast_forward_batches = None

  def _prepare_ema(self):
    if self.config.training.ema > 0:
      self.ema = create_ema(self._get_parameters(), decay=self.config.training.ema)
    else:
      self.ema = None

  def _validate_configuration(self):
    if self.config.algo.parameterization == 'ar':
      assert not self.config.algo.time_conditioning
      assert self.config.prior.type == 'none'

    if self.parameterization in {'score', 'mean'}:
      assert self.time_conditioning
    if self.T > 0:
      assert self.parameterization != 'score'

  def to(self, *args, **kwargs):
    self = super().to(*args, **kwargs) 
    self.metrics.to(*args, **kwargs)
    return self

  def q_xt(self, x, t):
    raise NotImplementedError
  
  def _get_parameters(self):
    return itertools.chain(self.backbone.parameters(), self.noise.parameters())

  def _eval_mode(self):
    if self.ema:
      self.ema.store(self._get_parameters())
      self.ema.copy_to(self._get_parameters())
    self.backbone.eval()
    self.noise.eval()

  def _train_mode(self):
    if self.ema:
      self.ema.restore(self._get_parameters())
    self.backbone.train()
    self.noise.train()

  def on_load_checkpoint(self, checkpoint):
    if self.ema:
      self.ema.load_state_dict(checkpoint['ema'])
    self.fast_forward_epochs = checkpoint['loops'][
      'fit_loop']['epoch_progress']['current']['completed']
    self.fast_forward_batches = checkpoint['loops'][
      'fit_loop']['epoch_loop.batch_progress'][
        'current']['completed']

  def on_save_checkpoint(self, checkpoint):
    if self.ema:
      checkpoint['ema'] = self.ema.state_dict()
    checkpoint['loops']['fit_loop'][
      'epoch_loop.batch_progress']['total'][
        'completed'] = checkpoint['loops']['fit_loop'][
          'epoch_loop.automatic_optimization.optim_progress'][
            'optimizer']['step']['total'][
              'completed'] * self.trainer.accumulate_grad_batches
    checkpoint['loops']['fit_loop'][
      'epoch_loop.batch_progress']['current'][
        'completed'] = checkpoint['loops']['fit_loop'][
          'epoch_loop.automatic_optimization.optim_progress'][
            'optimizer']['step']['current'][
              'completed'] * self.trainer.accumulate_grad_batches
    checkpoint['loops']['fit_loop'][
      'epoch_loop.state_dict'][
        '_batches_that_stepped'] = checkpoint['loops']['fit_loop'][
          'epoch_loop.automatic_optimization.optim_progress'][
            'optimizer']['step']['total']['completed']

  def on_train_start(self):
    if self.ema:
      self.ema.move_shadow_params_to_device(self.device)

  def optimizer_step(self, *args, **kwargs):
    super().optimizer_step(*args, **kwargs)
    if self.ema: self.ema.update(self._get_parameters())

  def _process_sigma(self, sigma):
    raise NotImplementedError

  def _process_model_output(self, model_output, xt, sigma):
    raise NotImplementedError

  def forward(self, xt, sigma, group_idxs=None):
    sigma = self._process_sigma(sigma)
    with torch.amp.autocast('cuda', dtype=torch.float32):
      if group_idxs is None:
        model_output = self.backbone(xt, sigma)
      else:
        model_output = self.backbone(xt, group_idxs, sigma)
    return self._process_model_output(model_output=model_output, xt=xt, sigma=sigma)

  def on_validation_batch_start(self, batch, batch_idx, dataloader_idx=0):
    self._val_cache = None

  def _loss(self, x0, valid_tokens,
            current_accumulation_step=None,
            train_mode=False):
    """Generic loss aggregation for all trainer modules."""
    input_tokens, valid_tokens = self._process_model_input(x0, valid_tokens)
    
    # Pass valid_tokens as keyword arg to avoid signature mismatch
    loss = self.nll(input_tokens, current_accumulation_step, train_mode, valid_tokens=valid_tokens)
    assert loss.ndim == 2
    
    if self.ignore_bos:
      loss[:, 0] = 0
      valid_tokens[:, 0] = 0
    if (getattr(self, 'shift_loss_targets', False)
        and valid_tokens.size(-1) == loss.size(-1) + 1):
      valid_tokens = valid_tokens[:, 1:]

    nlls = (loss * valid_tokens).sum()
    num_tokens = valid_tokens.sum()
    token_nll = nlls / num_tokens

    return Loss(loss=token_nll,
                nlls=nlls,
                num_tokens=num_tokens)

  def on_train_epoch_start(self):
    self.metrics.reset()
    assert self.metrics.train_nlls.nll.mean_value == 0
    assert self.metrics.train_nlls.nll.weight == 0

  def training_step(self, batch, batch_idx):
    current_accumulation_step = (
      batch_idx % self.trainer.accumulate_grad_batches)
    losses = self._loss(batch['input_ids'], batch['attention_mask'], current_accumulation_step, train_mode=True)
    self.metrics.update_train(losses.nlls, losses.num_tokens)
    self.log(name='trainer/loss', value=losses.loss, on_step=True, on_epoch=False, sync_dist=True, prog_bar=True)
    return losses.loss

  def on_validation_epoch_start(self):
    self.metrics.reset()
    self._eval_mode()
    assert self.metrics.valid_nlls.nll.mean_value == 0
    assert self.metrics.valid_nlls.nll.weight == 0

  def validation_step(self, batch, batch_idx):
    # Pass train_mode=False to log unweighted Full PPL and Denoising PPL
    _ = self._loss(batch['input_ids'], batch['attention_mask'], train_mode=False)
    
    # Standard optimization loss aggregation
    losses = self._loss(batch['input_ids'], batch['attention_mask'], train_mode=True)
    return losses.loss

  def on_validation_epoch_end(self):
    valid_metrics = {}
    for k, v in self.metrics.valid_nlls.items():
      if getattr(v, 'weight', 0) > 0:
        valid_metrics[k] = v.compute()
    if valid_metrics:
      self.log_dict(valid_metrics, on_step=False, on_epoch=True, sync_dist=True)
    
    # Log denoising metrics collection
    valid_metrics_denoising = {}
    for k, v in self.metrics.valid_nlls_denoising.items():
      if getattr(v, 'weight', 0) > 0:
        valid_metrics_denoising[k] = v.compute()
    if valid_metrics_denoising:
      self.log_dict(valid_metrics_denoising, on_step=False, on_epoch=True, sync_dist=True)

    if hasattr(self.metrics, 'valid_aux') and self.metrics.valid_aux.weight > 0:
      self.log(name='val/aux', value=self.metrics.valid_aux.compute(), on_step=False, on_epoch=True, sync_dist=True)
    if ((self.config.eval.compute_perplexity_on_sanity
         or not self.trainer.sanity_checking)
         and self.config.eval.generate_samples):
      try:
        samples, text_samples = None, None
        for _ in range(self.config.sampling.num_sample_batches):
          samples = self.generate_samples(num_samples=self.config.loader.eval_batch_size)
          self.metrics.record_entropy(samples)
          text_samples = self.tokenizer.batch_decode(samples, clean_up_tokenization_spaces=False)
        if text_samples is not None:
          if self.trainer.global_rank == 0 and hasattr(self.trainer.logger, 'log_table'):
            text_samples = text_samples[: self.config.sampling.num_sample_log]
            self.trainer.logger.log_table(
              key=f'samples@global_step{self.global_step}',
              columns=['Generated Samples'],
              data=[[s] for s in text_samples])
          self.log('val/sample_entropy', self.metrics.sample_entropy.compute(), on_epoch=True, on_step=False, sync_dist=True)
          if getattr(self.config.eval, 'save_validation_samples', False):
            save_dir = Path(os.getcwd()) / 'validation_samples'
            save_dir.mkdir(parents=True, exist_ok=True)
            save_path = save_dir / f'step_{self.global_step}.pt'
            torch.save(samples.detach().cpu(), save_path.as_posix())
      except Exception as e:
        print(f"Sampling failed at step {self.global_step}: {e}")
    self._train_mode()

  def configure_optimizers(self):
    optimizer = torch.optim.AdamW(
      self._get_parameters(),
      lr=self.config.optim.lr,
      betas=(self.config.optim.beta1,
             self.config.optim.beta2),
      eps=self.config.optim.eps,
      weight_decay=self.config.optim.weight_decay)
    scheduler = hydra.utils.instantiate(self.config.lr_scheduler, optimizer=optimizer)
    scheduler_dict = {'scheduler': scheduler, 'interval': 'step', 'monitor': 'val/loss', 'name': 'trainer/lr'}
    return [optimizer], [scheduler_dict]

  def _create_sampler(self):
    if self._sampler is not None:
      return self._sampler
    sampler_cfg = self._sampler_cfg
    if sampler_cfg is None:
      return None
    self._sampler = hydra.utils.instantiate(
      sampler_cfg,
      self.config,
      forward_process=getattr(self, '_forward_process', None),
      _recursive_=False,
    )
    return self._sampler

  def _resolve_sampler_config(self):
    algo_sampler = getattr(self.config.algo, 'sampler', None)
    if getattr(algo_sampler, '_target_', None):
      return algo_sampler
    sampling_sampler = getattr(self.config.sampling, 'sampler', None)
    if getattr(sampling_sampler, '_target_', None):
      return sampling_sampler
    return None

  @torch.no_grad()
  def generate_samples(self, num_samples, num_steps=None, eps=None):
    if num_steps is None:
      num_steps = self.config.sampling.steps
    if eps is None:
      eps = 1e-5
    inject_bos = getattr(self.config.sampling, 'inject_bos', True)
    sampler = self._create_sampler()
    if sampler is None:
      raise NotImplementedError(f"Algorithm {self.config.algo.name} has no sampler.")
    return sampler.generate(model=self, num_samples=num_samples, num_steps=num_steps, eps=eps, inject_bos=inject_bos)

  def _process_model_input(self, x0, valid_tokens):
    raise NotImplementedError

  def nll(self, input_tokens, current_accumulation_step=None, train_mode=False, valid_tokens=None):
    raise NotImplementedError

class Diffusion(TrainerBase):
  """Base class for diffusion-based algorithms."""
  def _call_nll_per_token(self, log_x_theta, xt, x0, alpha_t, dalpha_t, low_var, valid_tokens=None, train_mode=False):
    """Call nll_per_token with backward-compatible kwargs handling."""
    sig = inspect.signature(self.nll_per_token)
    params = sig.parameters
    kwargs = {'low_var': low_var}
    if 'valid_tokens' in params:
      kwargs['valid_tokens'] = valid_tokens
    if 'train_mode' in params:
      kwargs['train_mode'] = train_mode
    return self.nll_per_token(log_x_theta, xt, x0, alpha_t, dalpha_t, **kwargs)

  def _validate_configuration(self):
    super()._validate_configuration()
    assert self.loss_type in {'elbo', 'low_var'}

  def _process_model_input(self, x0, valid_tokens):
    return x0, valid_tokens

  def _get_jump_mask(self, xt, x0):
    return (xt != x0)

  def _metric_loss_unweighted(self, log_x_theta, xt, x0, alpha_t, dalpha_t):
    """Per-token quantity used for validation PPL-style metrics."""
    log_p_x0 = torch.gather(log_x_theta, -1, x0[:, :, None]).squeeze(-1)
    return -log_p_x0

  def _metric_loss_unweighted_denoising(self, log_x_theta, xt, x0, alpha_t, dalpha_t):
    """Per-token quantity used for denoising validation metrics."""
    return self._metric_loss_unweighted(log_x_theta, xt, x0, alpha_t, dalpha_t)

  def _metric_denoising_mask(self, xt, x0, jump_mask, valid_tokens):
    """Mask used for denoising validation metrics."""
    return jump_mask & valid_tokens.bool()

  def _q_xt_with_info(self, x0, t):
    """Return noisy tokens and optional process info dict.

    Prefers forward-process metadata (e.g. exact jump_mask / N_t>0 indicators)
    when the process exposes `return_info=True`.
    """
    fp = getattr(self, '_forward_process', None)
    if fp is not None:
      try:
        out = fp(x0, t, return_info=True)
        if isinstance(out, (tuple, list)):
          if len(out) >= 2 and isinstance(out[1], dict):
            return out[0], out[1]
          return out[0], {}
        return out, {}
      except TypeError:
        # Process does not support return_info; fall back to q_xt.
        pass
    return self.q_xt(x0, t), {}

  def nll(self, x0, current_accumulation_step=None, train_mode=False, valid_tokens=None):
    """Implements diffusion-style NLL evaluation with validation caching."""
    is_validation = not (hasattr(self, 'trainer') and self.trainer.training)
    
    if is_validation and self._val_cache is not None:
      if self._val_cache.get('x0_id') == x0.data_ptr():
        xt, sigma, log_x_theta, t = self._val_cache['xt'], self._val_cache['sigma'], self._val_cache['log_x_theta'], self._val_cache['t']
        alpha_t = self.noise.alpha_t(t).unsqueeze(-1)
        dalpha_t = self.noise.alpha_prime_t(t).unsqueeze(-1)
        return self._call_nll_per_token(
          log_x_theta, xt, x0, alpha_t, dalpha_t,
          low_var=train_mode and self.loss_type == 'low_var',
          valid_tokens=valid_tokens, train_mode=train_mode)

    t = self._sample_t(x0.shape[0], current_accumulation_step)
    if self.T > 0:
      t = (t * self.T).to(torch.int) / self.T + (1 / self.T)

    alpha_t = self.noise.alpha_t(t).unsqueeze(-1)
    dalpha_t = self.noise.alpha_prime_t(t).unsqueeze(-1)
    sigma = self._sigma_from_alphat(alpha_t)
    xt, fp_info = self._q_xt_with_info(x0, t)
    
    if getattr(self, 'shift_loss_targets', False):
      raw_logits = self.backbone(xt, sigma)
      raw_logits, x0, xt = utils.shift_for_next_token(raw_logits, x0, xt)
      ce = - raw_logits.log_softmax(-1).gather(-1, x0.unsqueeze(-1)).squeeze(-1)
      mask_positions = (xt == self.mask_id).to(ce.dtype)
      weighting = self.noise.alpha_prime_t(t).unsqueeze(-1) / (1 - alpha_t)
      return weighting * mask_positions * (-ce)
    else:
      log_x_theta = self.forward(xt, sigma=sigma)
      
      if is_validation:
        self._val_cache = {'x0_id': x0.data_ptr(), 'xt': xt, 't': t, 'sigma': sigma, 'log_x_theta': log_x_theta}

      # Metric Logging pass
      if not train_mode and valid_tokens is not None:
        metric_loss_full = self._metric_loss_unweighted(log_x_theta, xt, x0, alpha_t, dalpha_t)
        loss_unweighted_full = metric_loss_full * valid_tokens.to(metric_loss_full.dtype)
        self.metrics.update_valid(loss_unweighted_full.sum(), valid_tokens.sum())
        
        jump_mask = fp_info.get('jump_mask', self._get_jump_mask(xt, x0))
        jump_mask_valid = self._metric_denoising_mask(xt, x0, jump_mask, valid_tokens)
        num_jumped = jump_mask_valid.sum()
        if num_jumped > 0:
          metric_loss_denoising = self._metric_loss_unweighted_denoising(log_x_theta, xt, x0, alpha_t, dalpha_t)
          loss_unweighted_denoising = metric_loss_denoising * valid_tokens.to(metric_loss_denoising.dtype)
          loss_denoising = loss_unweighted_denoising * jump_mask_valid.to(loss_unweighted_denoising.dtype)
          self.metrics.update_valid_denoising(loss_denoising.sum(), num_jumped)

      return self._call_nll_per_token(
        log_x_theta, xt, x0, alpha_t, dalpha_t,
        low_var=train_mode and self.loss_type == 'low_var',
        valid_tokens=valid_tokens, train_mode=train_mode)

  def _process_sigma(self, sigma):
    sigma = sigma.mean(-1).squeeze()
    if sigma.ndim == 0: sigma = sigma.unsqueeze(0)
    if not self.time_conditioning: sigma = torch.zeros_like(sigma)
    return sigma

  def _sample_t(self, n, accum_step):
    if accum_step is not None:
      batch_dim = n
      n = self.config.loader.global_batch_size
    _eps_t = torch.rand(n, device=self.device)
    if self.antithetic_sampling:
      _eps_t = (_eps_t / n + torch.arange(n, device=self.device) / n) % 1
    t = (1 - self.sampling_eps) * _eps_t + self.sampling_eps
    if accum_step is not None:
      t = t.chunk(self.trainer.num_nodes)[self.trainer.node_rank]
      t = t.chunk(self.trainer.num_devices)[self.trainer.local_rank]
      t = t.chunk(self.trainer.accumulate_grad_batches)[accum_step]
      t = t[:batch_dim]
    return t

  def _sigma_from_alphat(self, alpha_t):
    return -torch.log(alpha_t)

  def _reconstruction_loss(self, x0):
    t0 = torch.zeros(1, x0.shape[0], dtype=self.dtype, device=self.device)
    sigma_t0 = self._sigma_from_alphat(self.noise.alpha_t(t0))
    model_output_t0 = self.forward(x0, sigma_t0)
    return -torch.gather(input=model_output_t0, dim=-1, index=x0[:, :, None]).squeeze(-1)

  def nll_per_token(self, model_output, xt, x0, alpha_t, dalpha_t, low_var, valid_tokens=None, train_mode=False):
    raise NotImplementedError

  def _get_score(self, x, sigma, group_idxs=None):
    raise NotImplementedError


class AbsorbingState(Diffusion):
  """Base class for absorbing state diffusion models."""
  def __init__(self, config, tokenizer):
    self.mask_id, vocab_size = ensure_mask_token(tokenizer)
    super().__init__(config, tokenizer, vocab_size=vocab_size)
    self.save_hyperparameters()
    fp_cfg = getattr(self.config.algo, 'forward_process', None)
    self._forward_process = hydra.utils.instantiate(omegaconf.OmegaConf.create(fp_cfg), tokenizer=self.tokenizer, schedule=self.noise, _recursive_=False)

  def _validate_configuration(self):
    super()._validate_configuration()
    if self.parameterization in {'score', 'mean'}: assert self.time_conditioning
    assert not (self.parameterization == 'mean' and self.T == 0)
    if self.T > 0: assert self.parameterization in {'mean', 'subs'}

  def q_xt(self, x, t):
    if not isinstance(t, torch.Tensor): t = torch.as_tensor(t, device=x.device, dtype=torch.float32)
    out = self._forward_process(x, t)
    xt = out[0] if isinstance(out, (tuple, list)) else out
    if self.ignore_bos: xt[:, 0] = x[:, 0]
    return xt

  def _get_jump_mask(self, xt, x0):
    # Fallback heuristic only. Prefer exact process-provided jump_mask when available.
    return (xt != x0)

  def prior_sample(self, *batch_dims):
    size = batch_dims[0] if len(batch_dims) == 1 and isinstance(batch_dims[0], (tuple, list)) else batch_dims
    return torch.full(tuple(size), self.mask_id, dtype=torch.int64, device=self.device)
