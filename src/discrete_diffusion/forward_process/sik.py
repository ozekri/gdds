"""Semantic-Informed Kernel (SIK) forward process.

Implements the SIK transition kernel as a specific instance of the 
Generalized CTMC Framework.
"""

from __future__ import annotations
import torch
from typing import Any, Optional

from .base_ctmc import BaseCTMCForwardProcess
from ..noise_schedules.base import NoiseSchedule
from .kernels.base import SIKKernel

class SIKForwardProcess(BaseCTMCForwardProcess):
    """SIK Implementation of the CTMC Framework.
    
    Implements the mixture kernel: F_t = (1-λ) * F_semantic + λ * Uniform.
    """

    def __init__(
        self,
        tokenizer,
        schedule: NoiseSchedule,
        kernel: SIKKernel,
        *,
        time_grid_size: int = 4096,
        time_eps: float = 1e-5,
        temperature_beta: float = 0.0,
        lambda_min: float = 0.01,
        lambda_sigmoid_s: float = 5.0,
        lambda_t0: float = 0.4,
        verbose: bool = False,
        name: Optional[str] = None,
    ) -> None:
        super().__init__(
            tokenizer=tokenizer, 
            schedule=schedule, 
            time_grid_size=time_grid_size, 
            time_eps=time_eps, 
            name=name or "sik"
        )
        self.kernel = kernel
        self.temperature_beta = temperature_beta
        self.lambda_min = lambda_min
        self.lambda_sigmoid_s = lambda_sigmoid_s
        self.lambda_t0 = lambda_t0
        self.verbose = verbose

    def _lambda(self, t: torch.Tensor) -> torch.Tensor:
        """λ(t): teleport probability."""
        s = self.lambda_sigmoid_s
        t0 = self.lambda_t0
        g = torch.sigmoid(s * (t - t0))
        device = t.device
        g0 = torch.sigmoid(torch.tensor(-s * t0, device=device, dtype=t.dtype))
        g1 = torch.sigmoid(torch.tensor(s * (1.0 - t0), device=device, dtype=t.dtype))
        g_norm = (g - g0) / (g1 - g0 + 1e-12)
        return self.lambda_min + (1.0 - self.lambda_min) * g_norm.clamp(0, 1)

    def _exponent(self, alpha_t: torch.Tensor) -> torch.Tensor:
        """Kernel exponent schedule used by SIK samplers and transitions."""
        return alpha_t.pow(self.temperature_beta)

    def transition_kernel(self, curr_tokens: torch.Tensor, tk: torch.Tensor) -> torch.Tensor:
        """Implements the SIK transition F_t."""
        device = curr_tokens.device
        
        # 1. Mixture logic: Semantic vs Uniform
        p_teleport = self._lambda(tk)
        is_uniform = torch.rand(tk.shape, device=device) < p_teleport
        
        # 2. Semantic branch (using standardized kernel API)
        alpha_tk = self.schedule.alpha_t(tk)
        exponent = self._exponent(alpha_tk)
        sem_next = self.kernel.sample_neighbors(curr_tokens, exponent)
        
        # 3. Uniform branch (rejection-free)
        uni_next = torch.randint(0, self.vocab_size - 1, tk.shape, device=device)
        uni_next = torch.where(uni_next >= curr_tokens, uni_next + 1, uni_next)
        
        return torch.where(is_uniform, uni_next, sem_next)

    def sample_prior(self, *batch_dims) -> torch.Tensor:
        size = batch_dims[0] if len(batch_dims) == 1 and isinstance(batch_dims[0], (tuple, list)) else batch_dims
        return torch.randint(low=0, high=self.vocab_size, size=tuple(size), dtype=torch.int64)

    def get_limiting_distribution(self) -> torch.Tensor:
        return torch.full((self.vocab_size,), 1.0 / float(self.vocab_size))


class GDDSGauss(SIKForwardProcess):
    """SIK Forward Process specifically using a Gaussian Kernel."""
    def __init__(self, *args, **kwargs):
        if "kernel" in kwargs and kwargs["kernel"].metric != "gaussian":
            print("[Warning] GDDSGauss initialized with non-gaussian kernel.")
        super().__init__(*args, **kwargs)
        self.name = "gdds_gauss"


class GDDSCosine(SIKForwardProcess):
    """SIK Forward Process specifically using a Cosine Kernel."""
    def __init__(self, *args, **kwargs):
        if "kernel" in kwargs and kwargs["kernel"].metric != "cosine":
            print("[Warning] GDDSCosine initialized with non-cosine kernel.")
        super().__init__(*args, **kwargs)
        self.name = "gdds_cosine"


__all__ = ["SIKForwardProcess", "GDDSGauss", "GDDSCosine"]
