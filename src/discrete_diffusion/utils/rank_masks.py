"""Utilities for rank-based attention masks."""

from __future__ import annotations

import torch

NEG_MASK = -1e4


def _ensure_batched(tensor: torch.Tensor) -> tuple[torch.Tensor, bool]:
    if tensor.dim() == 1:
        return tensor.unsqueeze(0), True
    return tensor, False


def _rank_pairs(rank: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    return rank[:, :, None], rank[:, None, :]


def _apply_pad_mask(mask: torch.Tensor, pad_mask: torch.Tensor | None) -> torch.Tensor:
    if pad_mask is None:
        return mask
    valid_q = ~pad_mask[:, :, None]
    valid_k = ~pad_mask[:, None, :]
    return mask & valid_q & valid_k


def _build_mask(
    rank: torch.Tensor,
    predicate,
    pad_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    ri, rj = _rank_pairs(rank)
    return _apply_pad_mask(predicate(rj, ri), pad_mask)


def compute_rank_from_tau(tau: torch.Tensor) -> torch.Tensor:
    """Convert times to ranks with deterministic tie-breaking."""
    tau, squeeze = _ensure_batched(tau)
    batch_size, seq_len = tau.shape
    del batch_size

    tau32 = tau.float()
    pos = torch.arange(seq_len, device=tau.device, dtype=torch.float32)
    pos = pos.unsqueeze(0).expand_as(tau32)
    tau_with_tiebreak = tau32 - 1e-6 * (pos / max(seq_len - 1, 1.0))

    perm = torch.argsort(tau_with_tiebreak, dim=1, descending=True)
    rank = torch.empty_like(perm, dtype=torch.long)
    rank.scatter_(
        1,
        perm,
        torch.arange(seq_len, device=tau.device, dtype=torch.long).unsqueeze(0).expand_as(perm),
    )
    return rank.squeeze(0) if squeeze else rank


def build_content_self_attn_mask(
    rank: torch.Tensor,
    pad_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Mask for rank-causal self-attention."""
    return _build_mask(rank, lambda rj, ri: rj <= ri, pad_mask)


def build_obs_self_attn_mask(
    rank: torch.Tensor,
    pad_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Mask for reverse-rank self-attention."""
    return _build_mask(rank, lambda rj, ri: rj >= ri, pad_mask)


def build_q_to_c_mask(
    rank: torch.Tensor,
    pad_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Strict query-to-content mask."""
    return _build_mask(rank, lambda rj, ri: rj < ri, pad_mask)


def build_fusion_q_to_o_mask(
    rank: torch.Tensor,
    pad_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Query-to-observation mask."""
    return _build_mask(rank, lambda rj, ri: rj >= ri, pad_mask)


def normalize_attn_mask(attn_mask: torch.Tensor | None, mode: str = "additive") -> torch.Tensor | None:
    """Convert masks to either boolean or additive SDPA format."""
    if attn_mask is None:
        return None
    if mode == "bool":
        return attn_mask if attn_mask.dtype == torch.bool else attn_mask > (NEG_MASK / 2)
    if mode == "additive":
        bool_mask = attn_mask if attn_mask.dtype == torch.bool else attn_mask > (NEG_MASK / 2)
        return torch.where(bool_mask, 0.0, NEG_MASK)
    raise ValueError(f"Unknown mode: {mode}. Expected 'bool' or 'additive'.")


def verify_no_self_in_q_to_c_mask(q_to_c_mask: torch.Tensor) -> bool:
    """Return True when the strict content mask excludes the diagonal."""
    return not torch.diagonal(q_to_c_mask, dim1=1, dim2=2).any()


def verify_self_in_fusion_mask(fusion_mask: torch.Tensor) -> bool:
    """Return True when the observation mask keeps the diagonal."""
    return torch.diagonal(fusion_mask, dim1=1, dim2=2).all()


def verify_mask_correctness(
    content_mask: torch.Tensor,
    obs_mask: torch.Tensor,
    q_to_c_mask: torch.Tensor,
    fusion_mask: torch.Tensor,
    rank: torch.Tensor,
) -> dict:
    """Check the expected rank relations for each mask."""
    ri, rj = _rank_pairs(rank)
    return {
        "content_is_rank_causal": (content_mask == (rj <= ri)).all().item(),
        "obs_is_reverse_rank": (obs_mask == (rj >= ri)).all().item(),
        "q_to_c_is_strict": (q_to_c_mask == (rj < ri)).all().item(),
        "q_to_c_no_self": verify_no_self_in_q_to_c_mask(q_to_c_mask),
        "fusion_is_reverse_rank": (fusion_mask == (rj >= ri)).all().item(),
        "fusion_has_self": verify_self_in_fusion_mask(fusion_mask),
    }


__all__ = [
    "compute_rank_from_tau",
    "build_content_self_attn_mask",
    "build_obs_self_attn_mask",
    "build_q_to_c_mask",
    "build_fusion_q_to_o_mask",
    "normalize_attn_mask",
    "verify_no_self_in_q_to_c_mask",
    "verify_self_in_fusion_mask",
    "verify_mask_correctness",
]
