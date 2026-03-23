"""Base framework for Continuous-Time Markov Chain (CTMC) noising.

Implements the general noising procedure described in Algorithm 1 of 
"Generalized Discrete Diffusion from Snapshots".
"""

from __future__ import annotations
import torch
from typing import Any, Optional, Tuple, Union

from .base import ForwardProcess
from ..noise_schedules.base import NoiseSchedule
from .kernels.base import SIKKernel

class BaseCTMCForwardProcess(ForwardProcess):
    """The General CTMC Framework (Algorithm 1).
    
    This class handles the temporal physics of the jump process:
    1. Sample number of jumps N_t ~ Poisson(Λ(t)).
    2. Sample exact non-homogeneous jump times T_1 < ... < T_N via inverse-CDF.
    3. Iterate through jumps and call the subclass 'transition_kernel'.
    """

    def __init__(
        self,
        tokenizer,
        schedule: NoiseSchedule,
        *,
        time_grid_size: int = 4096,
        time_eps: float = 1e-5,
        name: Optional[str] = None,
    ) -> None:
        super().__init__(tokenizer=tokenizer, schedule=schedule, name=name or "ctmc")
        self.vocab_size = len(tokenizer)
        self.time_grid_size = time_grid_size
        self.time_eps = time_eps
        
        # Pre-computed grids for inverse-CDF temporal sampling
        self._cdf_computed = False
        self._t_grid: Optional[torch.Tensor] = None
        self._Lambda_grid: Optional[torch.Tensor] = None

    def _precompute_inverse_cdf(self, device: torch.device):
        """Precompute lookup table for Λ(t) = -log(α(t))."""
        if self._cdf_computed and self._t_grid.device == device:
            return
        
        t_grid = torch.linspace(0, 1 - self.time_eps, self.time_grid_size, device=device)
        with torch.no_grad():
            alpha_grid = self.schedule.alpha_t(t_grid).clamp(min=1e-10)
            Lambda_grid = -torch.log(alpha_grid)
        
        self._t_grid = t_grid
        self._Lambda_grid = Lambda_grid
        self._cdf_computed = True

    def _Lambda(self, t: torch.Tensor) -> torch.Tensor:
        """Integrated intensity Λ(t) = -log(α(t))."""
        alpha = self.schedule.alpha_t(t).clamp(min=1e-10)
        return -torch.log(alpha)

    def _inverse_Lambda(self, s: torch.Tensor, device: torch.device) -> torch.Tensor:
        """Map values s through Λ⁻¹ by linear interpolation on the grid."""
        self._precompute_inverse_cdf(device)
        
        # Find indices in Lambda_grid
        indices = torch.searchsorted(self._Lambda_grid, s.reshape(-1), right=False)
        indices = torch.clamp(indices, 0, len(self._Lambda_grid) - 2)
        
        # Linear interpolation between grid points
        L0 = self._Lambda_grid[indices]; L1 = self._Lambda_grid[indices + 1]
        t0 = self._t_grid[indices]; t1 = self._t_grid[indices + 1]
        
        denom = (L1 - L0).clamp(min=1e-10)
        t = t0 + (s.reshape(-1) - L0) * (t1 - t0) / denom
        return t.reshape(s.shape)

    def transition_kernel(self, current_tokens: torch.Tensor, jump_times: torch.Tensor) -> torch.Tensor:
        """Sample next states x_k ~ F_{T_k}(· | x_{k-1}).
        
        Must be implemented by specific CTMC variants (e.g., SIK).
        """
        raise NotImplementedError

    @torch.no_grad()
    def forward(
        self,
        input_ids: torch.Tensor,
        t: torch.Tensor,
        return_info: bool = False,
        return_history: bool = False,
    ) -> torch.Tensor | Tuple[torch.Tensor, dict]:
        """Compute x_t ~ q(x_t | x_0) via Algorithm 1."""
        B, n = input_ids.shape
        device = input_ids.device
        if t.dim() == 1: t = t[:, None].expand(B, n)
        
        # Line 2: Sample N_t ~ Poisson(Λ(t))
        Lambda_t = self._Lambda(t)
        N_t = torch.poisson(Lambda_t).long()
        max_jumps = int(N_t.max().item())
        
        flat_input = input_ids.reshape(-1)
        flat_N = N_t.reshape(-1)
        flat_Lambda = Lambda_t.reshape(-1)
        current = flat_input.clone()
        M = flat_input.shape[0]
        
        history = [input_ids.clone()] if return_history else None

        if max_jumps == 0:
            info = {"num_jumps": N_t, "jump_mask": (N_t > 0), "history": history}
            return (input_ids, info) if return_info or return_history else input_ids

        # Line 3: Sample and sort jump times T_1 < ... < T_N
        # We sample s ~ Unif(0, Λ(t)) and map through Λ⁻¹
        rand = torch.rand(M, max_jumps, device=device)
        s_vals = rand * flat_Lambda.unsqueeze(1)
        
        T_samples = self._inverse_Lambda(s_vals, device).reshape(M, max_jumps)
        T_samples, _ = torch.sort(T_samples, dim=1)
        
        # Mask times beyond N_t
        valid = torch.arange(max_jumps, device=device)[None, :] < flat_N.unsqueeze(1)
        T_samples = torch.where(valid, T_samples, torch.full_like(T_samples, float("inf")))

        # Lines 4-6: Iterative Jump Loop
        for k in range(max_jumps):
            mask = (flat_N > k)
            if not mask.any(): break
            
            tk = T_samples[mask, k]
            curr_tokens = current[mask]
            
            # Subclass provides the specific transition probabilities F_t
            next_tokens = self.transition_kernel(curr_tokens, tk)
            
            current[mask] = next_tokens
            if return_history: history.append(current.reshape(B, n).clone())

        x_t = current.reshape(B, n)
        if return_info or return_history:
            info = {
                "num_jumps": N_t, 
                "jump_mask": (N_t > 0), 
                "jump_times": T_samples.reshape(B, n, max_jumps),
                "history": history
            }
            return x_t, info
        return x_t
