"""Campbell samplers."""

from __future__ import annotations

import torch
import torch.nn.functional as F

from ..forward_process.utils import sample_categorical
from ..utils.rank_masks import compute_rank_from_tau
from .base import Sampler


def _sampling_config(config):
    return config.sampling if hasattr(config, "sampling") else config


def _sequence_length(backbone) -> int:
    model_cfg = backbone.config.model if hasattr(backbone.config, "model") else backbone.config
    return model_cfg.length


def _expand_prompt(prompt_tokens, num_samples: int, seq_len: int) -> tuple[torch.Tensor | None, int]:
    if prompt_tokens is None:
        return None, 0
    prompt_len = prompt_tokens.shape[1]
    if prompt_len > seq_len:
        raise ValueError(f"Prompt length ({prompt_len}) exceeds sequence length ({seq_len})")
    if prompt_tokens.shape[0] == 1:
        prompt_tokens = prompt_tokens.expand(num_samples, -1)
    return prompt_tokens, prompt_len


def _get_special_token_ids(model, require_pad=True, require_mask=False):
    """Resolve pad and mask token ids from trainer, tokenizer, or backbone."""
    backbone = model.backbone
    pad_id = None
    mask_id = None

    if hasattr(model, "tokenizer") and model.tokenizer is not None:
        pad_id = model.tokenizer.pad_token_id
        mask_id = getattr(model.tokenizer, "mask_token_id", None)

    if pad_id is None and hasattr(model, "pad_token_id"):
        pad_id = model.pad_token_id
    if mask_id is None and hasattr(model, "mask_id"):
        mask_id = model.mask_id

    if pad_id is None and hasattr(backbone, "pad_token_id"):
        pad_id = backbone.pad_token_id
    if mask_id is None and hasattr(backbone, "mask_token_id"):
        mask_id = backbone.mask_token_id

    if require_pad:
        if pad_id is None:
            raise ValueError("Could not find pad_token_id.")
        if not 0 <= pad_id < backbone.vocab_size:
            raise ValueError(f"pad_token_id ({pad_id}) must be in [0, {backbone.vocab_size})")

    if require_mask:
        if mask_id is None:
            raise ValueError("Could not find mask_token_id.")
        if not 0 <= mask_id < backbone.vocab_size:
            raise ValueError(f"mask_token_id ({mask_id}) must be in [0, {backbone.vocab_size})")

    return pad_id, mask_id


class _CampbellSamplerBase(Sampler):
    def __init__(self, config, forward_process=None):
        self.config = config
        self.forward_process = forward_process

        sampling_cfg = _sampling_config(config)
        self.block_size = getattr(sampling_cfg, "perm_batch_size", 16)
        self.temperature = getattr(sampling_cfg, "temperature", 1.0)
        self.top_k = getattr(sampling_cfg, "top_k", 0)
        self.top_p = getattr(sampling_cfg, "top_p", 1.0)
        self.perm_progressive = getattr(sampling_cfg, "perm_progressive", False)
        self.perm_progressive_thresholds = getattr(sampling_cfg, "perm_progressive_thresholds", [64, 256])
        self.perm_progressive_blocks = getattr(sampling_cfg, "perm_progressive_blocks", [1, 4, 16])

    def _get_progressive_block_size(self, num_revealed: int, seq_len: int) -> int:
        thresholds = self.perm_progressive_thresholds
        blocks = self.perm_progressive_blocks

        if thresholds and max(thresholds) <= 1.0:
            thresholds = [int(value * seq_len) for value in thresholds]

        for index, threshold in enumerate(thresholds):
            if num_revealed < threshold:
                return blocks[index] if index < len(blocks) else blocks[-1]
        return blocks[-1] if blocks else self.block_size

    def _apply_sampling_params(self, logits: torch.Tensor) -> torch.Tensor:
        logits = logits.float()

        if self.temperature != 1.0:
            logits = logits / self.temperature

        if self.top_k > 0:
            top_k = min(self.top_k, logits.shape[-1])
            threshold = torch.topk(logits, top_k, dim=-1).values[..., -1, None]
            logits = logits.masked_fill(logits < threshold, float("-inf"))

        if self.top_p < 1.0:
            sorted_logits, sorted_indices = torch.sort(logits, descending=True, dim=-1)
            cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
            sorted_indices_to_remove = cumulative_probs > self.top_p
            sorted_indices_to_remove[..., 0] = False
            indices_to_remove = sorted_indices_to_remove.scatter(-1, sorted_indices, sorted_indices_to_remove)
            logits = torch.where(indices_to_remove, float("-inf"), logits)

        return torch.softmax(logits, dim=-1)

    def _sample_active_tokens(self, backbone, x_content, x_obs, tau, rank, active_mask):
        logits = backbone.forward_sampling(
            xcontent_ids=x_content,
            xobs_ids=x_obs,
            tau=tau,
            rank=rank,
        )
        active_logits = logits[active_mask].float()
        probs = self._apply_sampling_params(active_logits)
        sampled_tokens = sample_categorical(probs)
        x_content[active_mask] = sampled_tokens
        x_obs[active_mask] = sampled_tokens


class CampbellSamplerAbsorbing(_CampbellSamplerBase):
    """Serial permutation sampler for absorbing diffusion."""

    @torch.no_grad()
    def generate(
        self,
        model,
        *,
        num_samples,
        num_steps=None,
        eps=1e-5,
        inject_bos=None,
        prompt_tokens=None,
    ):
        del num_steps, eps, inject_bos
        device = model.device
        backbone = model.backbone
        seq_len = _sequence_length(backbone)
        pad_id, mask_id = _get_special_token_ids(model, require_pad=True, require_mask=True)

        x_content = torch.full((num_samples, seq_len), pad_id, dtype=torch.long, device=device)
        x_obs = torch.full((num_samples, seq_len), mask_id, dtype=torch.long, device=device)
        prompt_mask = torch.zeros(num_samples, seq_len, dtype=torch.bool, device=device)

        prompt_tokens, prompt_len = _expand_prompt(prompt_tokens, num_samples, seq_len)
        if prompt_tokens is not None:
            x_content[:, :prompt_len] = prompt_tokens
            x_obs[:, :prompt_len] = prompt_tokens
            prompt_mask[:, :prompt_len] = True

        tau = torch.rand(num_samples, seq_len, device=device)
        if prompt_len > 0:
            tau[:, :prompt_len] = 2.0
        rank = compute_rank_from_tau(tau)

        for step_idx in range(seq_len):
            active_mask = (rank == step_idx) & (~prompt_mask)
            if active_mask.any():
                self._sample_active_tokens(backbone, x_content, x_obs, tau, rank, active_mask)

        return x_content


class CampbellSamplerAbsorbingBatched(_CampbellSamplerBase):
    """Batched permutation sampler for absorbing diffusion."""

    @torch.no_grad()
    def generate(
        self,
        model,
        *,
        num_samples,
        num_steps=None,
        eps=1e-5,
        inject_bos=None,
        prompt_tokens=None,
    ):
        del num_steps, eps, inject_bos
        device = model.device
        backbone = model.backbone
        seq_len = _sequence_length(backbone)
        pad_id, mask_id = _get_special_token_ids(model, require_pad=True, require_mask=True)

        x_content = torch.full((num_samples, seq_len), pad_id, dtype=torch.long, device=device)
        x_obs = torch.full((num_samples, seq_len), mask_id, dtype=torch.long, device=device)
        prompt_mask = torch.zeros(num_samples, seq_len, dtype=torch.bool, device=device)

        prompt_tokens, prompt_len = _expand_prompt(prompt_tokens, num_samples, seq_len)
        if prompt_tokens is not None:
            x_content[:, :prompt_len] = prompt_tokens
            x_obs[:, :prompt_len] = prompt_tokens
            prompt_mask[:, :prompt_len] = True

        tau = torch.rand(num_samples, seq_len, device=device)
        if prompt_len > 0:
            tau[:, :prompt_len] = 2.0
        rank = compute_rank_from_tau(tau)
        known_mask = prompt_mask.clone()

        if self.perm_progressive:
            batch_idx = 0
            while batch_idx < seq_len * 2 and not known_mask.all():
                num_revealed = int(known_mask.float().sum().item() / num_samples)
                current_block_size = self._get_progressive_block_size(num_revealed, seq_len)
                unprocessed_mask = (~prompt_mask) & (~known_mask)
                if not unprocessed_mask.any():
                    break

                min_unprocessed_rank = rank[unprocessed_mask].min().item()
                active_mask = (
                    unprocessed_mask
                    & (rank >= min_unprocessed_rank)
                    & (rank < min_unprocessed_rank + current_block_size)
                )
                if active_mask.any():
                    self._sample_active_tokens(backbone, x_content, x_obs, tau, rank, active_mask)
                    known_mask |= active_mask
                batch_idx += 1
        else:
            num_batches = (seq_len + self.block_size - 1) // self.block_size
            for batch_idx in range(num_batches):
                rank_start = batch_idx * self.block_size
                rank_end = min((batch_idx + 1) * self.block_size, seq_len)
                active_mask = (rank >= rank_start) & (rank < rank_end) & (~prompt_mask)
                if active_mask.any():
                    self._sample_active_tokens(backbone, x_content, x_obs, tau, rank, active_mask)

        return x_content


class CampbellSamplerUniform(_CampbellSamplerBase):
    """Permutation sampler for uniform CTMC diffusion."""

    @torch.no_grad()
    def generate(
        self,
        model,
        *,
        num_samples,
        num_steps=None,
        eps=1e-5,
        inject_bos=None,
        prompt_tokens=None,
    ):
        del num_steps, eps, inject_bos
        device = model.device
        backbone = model.backbone
        seq_len = _sequence_length(backbone)
        vocab_size = backbone.vocab_size
        pad_id, _ = _get_special_token_ids(model, require_pad=True, require_mask=False)

        x_content = torch.full((num_samples, seq_len), pad_id, dtype=torch.long, device=device)
        x_obs = torch.randint(0, vocab_size, (num_samples, seq_len), dtype=torch.long, device=device)
        prompt_mask = torch.zeros(num_samples, seq_len, dtype=torch.bool, device=device)

        prompt_tokens, prompt_len = _expand_prompt(prompt_tokens, num_samples, seq_len)
        if prompt_tokens is not None:
            x_content[:, :prompt_len] = prompt_tokens
            x_obs[:, :prompt_len] = prompt_tokens
            prompt_mask[:, :prompt_len] = True

        tau = torch.rand(num_samples, seq_len, device=device)
        if prompt_len > 0:
            tau[:, :prompt_len] = 2.0
        rank = compute_rank_from_tau(tau)
        known_mask = prompt_mask.clone()

        if self.perm_progressive:
            batch_idx = 0
            while batch_idx < seq_len * 2 and not known_mask.all():
                num_revealed = int(known_mask.float().sum().item() / num_samples)
                current_block_size = self._get_progressive_block_size(num_revealed, seq_len)
                unprocessed_mask = (~prompt_mask) & (~known_mask)
                if not unprocessed_mask.any():
                    break

                min_unprocessed_rank = rank[unprocessed_mask].min().item()
                active_mask = (
                    unprocessed_mask
                    & (rank >= min_unprocessed_rank)
                    & (rank < min_unprocessed_rank + current_block_size)
                )
                if active_mask.any():
                    self._sample_active_tokens(backbone, x_content, x_obs, tau, rank, active_mask)
                    known_mask |= active_mask
                batch_idx += 1
        else:
            num_batches = (seq_len + self.block_size - 1) // self.block_size
            for batch_idx in range(num_batches):
                rank_start = batch_idx * self.block_size
                rank_end = min((batch_idx + 1) * self.block_size, seq_len)
                active_mask = (rank >= rank_start) & (rank < rank_end) & (~prompt_mask)
                if active_mask.any():
                    self._sample_active_tokens(backbone, x_content, x_obs, tau, rank, active_mask)

        return x_content


CampbellSampler = CampbellSamplerAbsorbingBatched


__all__ = [
    "CampbellSampler",
    "CampbellSamplerAbsorbing",
    "CampbellSamplerAbsorbingBatched",
    "CampbellSamplerUniform",
]
