"""Lightning callbacks used across discrete_diffusion."""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass
from typing import Dict

import lightning as L
import numpy as np
import torch
from lightning.pytorch.utilities import rank_zero_only
from rich.console import Console
from rich.table import Table


@dataclass
class _TimingStats:
  latency_mean: float
  latency_std: float
  throughput_mean: float
  throughput_std: float


class TrainingLatencyCallback(L.Callback):
  """Measure forward/backward latency on synthetic data during training.

  The callback generates synthetic batches once the configured ``start_step``
  is reached and records latency/throughput statistics without touching the
  real dataloader. Metrics are logged once and saved under ``train_latency``
  in the current working directory.
  """

  def __init__(
      self,
      enabled: bool = False,
      start_step: int = 500,
      num_batches: int = 500,
      num_warmup: int = 10,
      batch_size: int | None = None,
      sequence_length: int | None = None,
      save_name: str = "callback_latency.json",
      pretty_print: bool = True) -> None:
    super().__init__()
    self.enabled = enabled
    self.start_step = start_step
    self.num_batches = num_batches
    self.num_warmup = num_warmup
    self.batch_size = batch_size
    self.sequence_length = sequence_length
    self.save_name = save_name
    self.pretty_print = pretty_print
    self._has_run = False

  def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):  # noqa: D401
    if not self.enabled or self._has_run:
      return

    if trainer.global_step < self.start_step:
      return

    strategy = trainer.strategy
    if strategy is not None:
      strategy.barrier()

    results: Dict[str, Dict[str, float]] | None = None
    if getattr(trainer, 'global_rank', 0) == 0:
      results = self._run_measurement(trainer, pl_module)
      self._log_results(trainer, results)

    self._has_run = True

    if strategy is not None:
      strategy.barrier()

  def _run_measurement(self, trainer, pl_module) -> Dict[str, Dict[str, float]]:
    device = pl_module.device
    batch_size = self.batch_size or getattr(
      pl_module.config.loader, 'batch_size', None)
    if batch_size is None:
      raise ValueError('`batch_size` must be provided for latency timing.')

    sequence_length = self.sequence_length or getattr(
      pl_module.config.model, 'length', None)
    if sequence_length is None:
      raise ValueError('`sequence_length` must be provided for latency timing.')

    vocab_size = getattr(pl_module, 'vocab_size', None)
    if vocab_size is None:
      if hasattr(pl_module, 'tokenizer'):
        vocab_size = len(pl_module.tokenizer)
      else:
        raise ValueError('Could not infer vocabulary size for latency timing.')

    forward_stats = self._time_forward(
      pl_module, vocab_size, batch_size, sequence_length, device)
    fwd_bwd_stats = self._time_forward_backward(
      pl_module, vocab_size, batch_size, sequence_length, device)

    return {
      'forward': forward_stats.__dict__,
      'forward_backward': fwd_bwd_stats.__dict__,
      'meta': {
        'start_step': self.start_step,
        'measured_step': trainer.global_step,
        'batch_size': batch_size,
        'sequence_length': sequence_length,
        'num_warmup': self.num_warmup,
        'num_batches': self.num_batches,
      }
    }

  def _time_forward(self, pl_module, vocab_size, batch_size,
                    sequence_length, device) -> _TimingStats:
    latencies, throughputs = self._measure(
      pl_module, vocab_size, batch_size, sequence_length, device,
      backward=False)
    return self._aggregate(latencies, throughputs)

  def _time_forward_backward(self, pl_module, vocab_size, batch_size,
                             sequence_length, device) -> _TimingStats:
    latencies, throughputs = self._measure(
      pl_module, vocab_size, batch_size, sequence_length, device,
      backward=True)
    return self._aggregate(latencies, throughputs)

  def _measure(self, pl_module, vocab_size, batch_size,
               sequence_length, device, backward):
    total_iters = self.num_warmup + self.num_batches
    latencies = []
    throughputs = []

    for _ in range(total_iters):
      x0 = torch.randint(
        0, vocab_size,
        size=(batch_size, sequence_length),
        device=device)
      valid_tokens = torch.ones_like(x0, device=device)

      if device.type == 'cuda' and torch.cuda.is_available():
        torch.cuda.synchronize(device)
      start_time = time.perf_counter()
      losses = pl_module._loss(x0, valid_tokens, train_mode=True)
      if backward:
        losses.loss.backward()
        pl_module.zero_grad(set_to_none=True)
      if device.type == 'cuda' and torch.cuda.is_available():
        torch.cuda.synchronize(device)
      end_time = time.perf_counter()

      latency = end_time - start_time
      latencies.append(latency)
      throughputs.append((batch_size * sequence_length) / latency)

    latencies = latencies[self.num_warmup:]
    throughputs = throughputs[self.num_warmup:]
    return latencies, throughputs

  def _aggregate(self, latencies, throughputs) -> _TimingStats:
    latency_arr = np.asarray(latencies, dtype=np.float64)
    throughput_arr = np.asarray(throughputs, dtype=np.float64)
    return _TimingStats(
      latency_mean=float(latency_arr.mean()),
      latency_std=float(latency_arr.std(ddof=0)),
      throughput_mean=float(throughput_arr.mean()),
      throughput_std=float(throughput_arr.std(ddof=0)))

  @rank_zero_only
  def _log_results(self, trainer, results):
    if trainer.logger is not None:
      metrics = {
        'latency/forward_mean_s': results['forward']['latency_mean'],
        'latency/forward_std_s': results['forward']['latency_std'],
        'throughput/forward_mean_tokens_per_s':
          results['forward']['throughput_mean'],
        'throughput/forward_std_tokens_per_s':
          results['forward']['throughput_std'],
        'latency/forward_backward_mean_s':
          results['forward_backward']['latency_mean'],
        'latency/forward_backward_std_s':
          results['forward_backward']['latency_std'],
        'throughput/forward_backward_mean_tokens_per_s':
          results['forward_backward']['throughput_mean'],
        'throughput/forward_backward_std_tokens_per_s':
          results['forward_backward']['throughput_std'],
      }
      trainer.logger.log_metrics(metrics, step=trainer.global_step)

    save_dir = os.path.join(os.getcwd(), 'train_latency')
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, self.save_name)
    with open(save_path, 'w') as fp:
      json.dump(results, fp, indent=4)

    if self.pretty_print:
      table = Table(title='Training Latency (synthetic batches)')
      table.add_column('Phase')
      table.add_column('Latency (s)', justify='right')
      table.add_column('Throughput (tokens/s)', justify='right')
      table.add_row(
        'Forward',
        f"{results['forward']['latency_mean']:.6f} ± "
        f"{results['forward']['latency_std']:.6f}",
        f"{results['forward']['throughput_mean']:.1f} ± "
        f"{results['forward']['throughput_std']:.1f}")
      table.add_row(
        'Forward+Backward',
        f"{results['forward_backward']['latency_mean']:.6f} ± "
        f"{results['forward_backward']['latency_std']:.6f}",
        f"{results['forward_backward']['throughput_mean']:.1f} ± "
        f"{results['forward_backward']['throughput_std']:.1f}")
      Console().print(table)
