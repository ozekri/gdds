"""Sampler interface for discrete diffusion generation routines."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Optional


class Sampler(ABC):
  """Base interface defining hooks used by all samplers.
  
  Samplers orchestrate the iterative generation process, managing the
  transition from noise to clean data.
  """

  @abstractmethod
  def generate(self, model: Any, *, num_samples: int, num_steps: int, eps: float,
               inject_bos: bool) -> Any:
    """Generate new samples from the provided model.
    
    Args:
        model: The trained model to sample from.
        num_samples: Number of samples to generate.
        num_steps: Number of sampling steps.
        eps: Small epsilon for numerical stability or time bounds.
        inject_bos: Whether to inject a Beginning-Of-Sequence token.
        
    Returns:
        Tensor: Generated samples.
    """
    raise NotImplementedError

  def compute_posterior(self, x: Any, t: Any, dt: Any, p_x0_cache: Optional[Any]) -> Any:
    """Optional posterior computation hook for samplers that need incremental steps."""
    raise NotImplementedError

  def step_analytic(self, x: Any, t: Any, dt: Any) -> Any:
    """Optional analytic update hook for samplers that support closed-form steps."""
    raise NotImplementedError

  def denoise(self, x: Any, t: Any) -> Any:
    """Optional denoiser update hook for samplers that clean up predictions."""
    raise NotImplementedError
