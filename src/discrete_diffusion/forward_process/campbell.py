"""Campbell event sampling built on top of the base CTMC framework."""

from __future__ import annotations

from typing import Literal

import torch

from .base_ctmc import BaseCTMCForwardProcess
from ..noise_schedules.base import NoiseSchedule
from ..utils.rank_masks import compute_rank_from_tau


class _CampbellCTMCForwardProcess(BaseCTMCForwardProcess):
    def sample_first_jump_times_exact(
        self,
        batch_size: int,
        seq_len: int,
        device: torch.device,
        t_horizon: float = 1.0,
        no_jump_sentinel: float = 2.0,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        self._precompute_inverse_cdf(device)

        rate = torch.full((batch_size, seq_len), self._Lambda(torch.tensor(t_horizon, device=device)).item(), device=device)
        num_jumps = torch.poisson(rate).long()
        has_jumped = num_jumps >= 1

        s = torch.rand(batch_size, seq_len, device=device, dtype=torch.float32) * rate
        tau = self._inverse_Lambda(s, device).clamp(min=0.0, max=t_horizon - self.time_eps)
        tau = torch.where(has_jumped, tau, torch.full_like(tau, no_jump_sentinel))
        return tau, has_jumped

    def sample_first_jump_times(
        self,
        batch_size: int,
        seq_len: int,
        device: torch.device,
    ) -> torch.Tensor:
        tau, _ = self.sample_first_jump_times_exact(batch_size, seq_len, device)
        return tau


class CampbellAbsorbingForwardProcess(_CampbellCTMCForwardProcess):
    """Absorbing Campbell forward process backed by the generic CTMC sampler."""

    def __init__(
        self,
        tokenizer,
        schedule: NoiseSchedule,
        *,
        mask_token_id: int,
        time_grid_size: int = 4096,
        time_eps: float = 1e-5,
        name: str | None = None,
    ) -> None:
        super().__init__(
            tokenizer=tokenizer,
            schedule=schedule,
            time_grid_size=time_grid_size,
            time_eps=time_eps,
            name=name or "campbell_absorbing",
        )
        self.mask_token_id = mask_token_id

    def transition_kernel(self, current_tokens: torch.Tensor, jump_times: torch.Tensor) -> torch.Tensor:
        del jump_times
        return torch.full_like(current_tokens, self.mask_token_id)

    def sample_observations(self, x0: torch.Tensor, tau: torch.Tensor, t: float = 1.0) -> torch.Tensor:
        mask_tokens = torch.full_like(x0, self.mask_token_id)
        return torch.where(tau <= t, mask_tokens, x0)


class CampbellUniformForwardProcess(_CampbellCTMCForwardProcess):
    """Uniform Campbell forward process backed by the generic CTMC sampler."""

    def __init__(
        self,
        tokenizer,
        schedule: NoiseSchedule,
        *,
        time_grid_size: int = 4096,
        time_eps: float = 1e-5,
        name: str | None = None,
    ) -> None:
        super().__init__(
            tokenizer=tokenizer,
            schedule=schedule,
            time_grid_size=time_grid_size,
            time_eps=time_eps,
            name=name or "campbell_uniform",
        )

    def transition_kernel(self, current_tokens: torch.Tensor, jump_times: torch.Tensor) -> torch.Tensor:
        del jump_times
        return torch.randint(0, self.vocab_size, current_tokens.shape, device=current_tokens.device, dtype=current_tokens.dtype)

    def sample_observations(self, x0: torch.Tensor, tau: torch.Tensor, t: float = 1.0) -> torch.Tensor:
        random_tokens = torch.randint(0, self.vocab_size, x0.shape, device=x0.device, dtype=x0.dtype)
        return torch.where(tau <= t, random_tokens, x0)

    def build_training_batch_multijump(
        self,
        x0: torch.Tensor,
        t: float = 1.0,
        max_jumps: int = 50,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        t_tensor = torch.full(x0.shape, t, device=x0.device, dtype=torch.float32)
        _, info = self.forward(x0, t_tensor, return_info=True, return_history=True)

        num_jumps = info["num_jumps"]
        history = info["history"]
        full_jump_times = info["jump_times"]

        jump_times = torch.full((*x0.shape, max_jumps), float("inf"), device=x0.device)
        used_jump_steps = min(full_jump_times.shape[-1], max_jumps)
        if used_jump_steps > 0:
            jump_times[:, :, :used_jump_steps] = full_jump_times[:, :, :used_jump_steps]

        max_history_steps = min(len(history) - 1, max_jumps)
        pre_jump_tokens = torch.zeros((*x0.shape, max_jumps), dtype=x0.dtype, device=x0.device)
        post_jump_tokens = torch.zeros_like(pre_jump_tokens)

        for jump_idx in range(max_history_steps):
            pre_jump_tokens[:, :, jump_idx] = history[jump_idx]
            post_jump_tokens[:, :, jump_idx] = history[jump_idx + 1]

        event_mask = torch.arange(max_jumps, device=x0.device).view(1, 1, max_jumps) < num_jumps.unsqueeze(-1)
        truncated_z_t = history[max_history_steps] if max_history_steps > 0 else history[0]
        return truncated_z_t, num_jumps, jump_times, pre_jump_tokens, post_jump_tokens, event_mask


class CampbellEventSampler:
    """Event-sampler adapter used by CampbellTrainer."""

    def __init__(
        self,
        schedule: NoiseSchedule,
        tokenizer,
        mode: Literal["absorbing", "uniform"] = "absorbing",
        mask_token_id: int | None = None,
        time_grid_size: int = 4096,
        time_eps: float = 1e-5,
    ) -> None:
        self.mode = mode
        self.mask_token_id = mask_token_id

        if mode == "absorbing":
            if mask_token_id is None:
                raise ValueError("mask_token_id required for absorbing mode")
            self.process = CampbellAbsorbingForwardProcess(
                tokenizer=tokenizer,
                schedule=schedule,
                mask_token_id=mask_token_id,
                time_grid_size=time_grid_size,
                time_eps=time_eps,
            )
        elif mode == "uniform":
            self.process = CampbellUniformForwardProcess(
                tokenizer=tokenizer,
                schedule=schedule,
                time_grid_size=time_grid_size,
                time_eps=time_eps,
            )
        else:
            raise ValueError(f"Unknown mode: {mode}. Use 'absorbing' or 'uniform'.")

    def sample_tau_exact(
        self,
        batch_size: int,
        seq_len: int,
        device: torch.device,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return self.process.sample_first_jump_times_exact(batch_size, seq_len, device)

    def sample_tau(
        self,
        batch_size: int,
        seq_len: int,
        device: torch.device,
    ) -> torch.Tensor:
        return self.process.sample_first_jump_times(batch_size, seq_len, device)

    def sample_z_obs(
        self,
        x0: torch.Tensor,
        tau: torch.Tensor,
        t: float = 1.0,
    ) -> torch.Tensor:
        return self.process.sample_observations(x0, tau, t)

    def sample(
        self,
        x0: torch.Tensor,
        tau: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        batch_size, seq_len = x0.shape
        if tau is None:
            tau = self.sample_tau(batch_size, seq_len, x0.device)
        z_obs = self.sample_z_obs(x0, tau)
        rank = compute_rank_from_tau(tau)
        return z_obs, tau, rank

    def build_training_batch_multijump(
        self,
        x0: torch.Tensor,
        t: float = 1.0,
        max_jumps: int = 50,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        if self.mode != "uniform":
            raise ValueError("Multi-jump training is only defined for uniform mode.")
        return self.process.build_training_batch_multijump(x0, t=t, max_jumps=max_jumps)


__all__ = [
    "CampbellAbsorbingForwardProcess",
    "CampbellUniformForwardProcess",
    "CampbellEventSampler",
]
