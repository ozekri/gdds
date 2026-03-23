"""Utilities for forward-process implementations.

Includes tokenizer helpers and a numerically stable categorical sampler.
"""

from __future__ import annotations

import torch


def _effective_vocab_size(tokenizer) -> int:
  """Return the effective vocabulary size for a tokenizer.
  
  Prefer an explicitly annotated `_effective_vocab_size`, otherwise fall back to
  the actual length (which reflects special tokens added via `add_special_tokens`).
  """
  eff = getattr(tokenizer, "_effective_vocab_size", None)
  if eff is not None:
    return int(eff)
  return int(len(tokenizer))


def _mask_token_id(tokenizer) -> int:
  """Return the tokenizer's mask token id, raising if not defined."""
  mask_id = getattr(tokenizer, "mask_token_id", None)
  if mask_id is None:
    raise ValueError("Mask token id is not defined for tokenizer")
  return int(mask_id)


def _unsqueeze(x: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
  """Match `x` rank to `reference` by appending singleton dims."""
  return x.view(*x.shape, * ((1,) * (len(reference.shape) - len(x.shape))))


def sample_categorical(categorical_probs: torch.Tensor) -> torch.Tensor:
  """Sample categories via a Gumbel-max formulation for stability.

  Expects `categorical_probs` to be non-negative and to sum to one along the
  last dimension. This implementation mirrors the stable sampler used in the
  existing absorbing helpers for consistency.
  """
  gumbel_norm = 1e-10 - (torch.rand_like(categorical_probs) + 1e-10).log()
  return (categorical_probs / gumbel_norm).argmax(dim=-1)

