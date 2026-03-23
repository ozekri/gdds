"""Lightning DataModule for discrete diffusion datasets."""

from __future__ import annotations

from typing import Optional, Tuple

import lightning as L

from .loaders import get_dataloaders


class DiscreteDiffusionDataModule(L.LightningDataModule):
  """Lightning DataModule for discrete diffusion datasets.
  
  Wraps dataset loading logic to provide train and validation dataloaders
  to the Lightning Trainer.
  """

  def __init__(self,
               config,
               tokenizer,
               *,
               skip_train: bool = False,
               skip_valid: bool = False,
               valid_seed: Optional[int] = None):
    """Initialize the DataModule.

    Args:
        config: Hydra configuration object.
        tokenizer: Tokenizer instance.
        skip_train: Whether to skip creating the training loader.
        skip_valid: Whether to skip creating the validation loader.
        valid_seed: Optional seed for validation set shuffling.
    """
    super().__init__()
    self.config = config
    self.tokenizer = tokenizer
    self.skip_train = skip_train
    self.skip_valid = skip_valid
    self.valid_seed = valid_seed
    self._train_loader = None
    self._valid_loader = None

  def _build_loaders(self) -> Tuple[Optional[object], Optional[object]]:
    return get_dataloaders(
      self.config,
      self.tokenizer,
      skip_train=self.skip_train,
      skip_valid=self.skip_valid,
      valid_seed=self.valid_seed)

  def setup(self, stage: Optional[str] = None):
    """Set up datasets for the given stage.
    
    Handles distributed data loading synchronization to ensure valid/test sets
    are consistent across ranks.
    """
    if stage not in (None, "fit", "validate"):
      return
    if self._train_loader is not None or self._valid_loader is not None:
      return

    trainer = getattr(self, "trainer", None)
    strategy = getattr(trainer, "strategy", None) if trainer else None
    barrier = getattr(strategy, "barrier", None) if strategy else None
    is_rank_zero = bool(getattr(trainer, "is_global_zero", True))

    if barrier is not None:
      if is_rank_zero:
        self._train_loader, self._valid_loader = self._build_loaders()
      barrier()
      if not is_rank_zero:
        self._train_loader, self._valid_loader = self._build_loaders()
      barrier()
    else:
      self._train_loader, self._valid_loader = self._build_loaders()

  def train_dataloader(self):
    if self._train_loader is None:
      raise RuntimeError("DataModule setup() must run before requesting train loader.")
    return self._train_loader

  def val_dataloader(self):
    if self._valid_loader is None:
      raise RuntimeError("DataModule setup() must run before requesting valid loader.")
    return self._valid_loader

