"""Periodic sample saving hook for discrete diffusion models."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import lightning as L
import torch
from omegaconf import OmegaConf


class SampleSaver(L.Callback):
  """Save generated tokens every ``every_n_steps`` during training."""

  def __init__(
      self,
      enabled: bool = False,
      every_n_steps: int = 1000,
      num_samples: Optional[int] = None,
      num_steps: Optional[int] = None,
      save_dir: str = './samples/',
      filename_template: str = 'step_{global_step}.json') -> None:
    super().__init__()
    if every_n_steps <= 0:
      raise ValueError('every_n_steps must be positive')

    self.enabled = enabled
    self.every_n_steps = every_n_steps
    self.num_samples = num_samples
    self.num_steps = num_steps
    self.save_dir = Path(save_dir)
    self.filename_template = filename_template

  def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
    del outputs, batch, batch_idx
    if not self.enabled or not trainer.is_global_zero:
      return

    global_step = max(1, trainer.global_step)
    if global_step % self.every_n_steps != 0:
      return

    samples = pl_module.generate_samples(
      num_samples=self._resolve_num_samples(pl_module),
      num_steps=self._resolve_num_steps(pl_module))
    samples = samples.detach().cpu()
    save_path = self._build_save_path(global_step)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    text_samples = pl_module.tokenizer.batch_decode(
        samples.tolist(),
        clean_up_tokenization_spaces=False  # Explicitly set to avoid FutureWarning
    )
    entropy = self._mean_entropy(samples)
    metadata = dict(
      text=text_samples,
      entropy=entropy,
      config=OmegaConf.to_container(pl_module.config, resolve=True),
    )
    with open(save_path, 'w', encoding='utf-8') as fp:
      json.dump(metadata, fp, indent=2)

  def _mean_entropy(self, samples: torch.Tensor) -> float:
    if samples.numel() == 0:
      return 0.0
    entropies = []
    for sample in samples.unbind(0):
      _, counts = torch.unique(sample, return_counts=True, sorted=False)
      probs = counts.float() / counts.sum()
      entropies.append(float(torch.special.entr(probs).sum()))
    return float(sum(entropies) / len(entropies))

  def _build_save_path(self, global_step: int) -> Path:
    filename = self.filename_template.format(global_step=global_step)
    return self.save_dir / filename

  def _resolve_num_samples(self, pl_module) -> int:
    if self.num_samples is not None:
      return self.num_samples
    batch_size = getattr(pl_module.config.loader, 'eval_batch_size', None)
    if batch_size is None:
      raise ValueError('Could not infer num_samples for SampleSaver')
    return batch_size

  def _resolve_num_steps(self, pl_module) -> int:
    if self.num_steps is not None:
      return self.num_steps
    steps = getattr(pl_module.config.sampling, 'steps', None)
    if steps is None:
      raise ValueError('Could not infer sampling steps for SampleSaver')
    return steps
