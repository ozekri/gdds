"""Exponential noise schedule: α(t) = exp(-λ t).

Then Λ(t) = -log(α(t)) = λ t is linear in t, so on a uniform t-grid both
Δt and ΔΛ are constant (Δt fixed by the grid, ΔΛ = λ Δt).
"""

from __future__ import annotations

import math

import torch

from .base import NoiseSchedule


class Exponential(NoiseSchedule):
    """Exponential schedule: α(t) = exp(-λ t).

    Λ(t) = λ t, so both Δt and ΔΛ are constant on a uniform t-grid:
    - Uniform t: t_k = 1 - k·Δt ⇒ constant Δt.
    - ΔΛ = Λ(t) - Λ(u) = λ(t - u) = λ Δt ⇒ constant ΔΛ.

    Parameterize via `lam` (rate) or `eps` (α(1) = eps ⇒ λ = -log(eps)).
    """

    def __init__(self, lam: float | None = None, eps: float = 1e-5, **kwargs):
        super().__init__()
        if lam is not None:
            self.lam = float(lam)
        else:
            self.lam = -math.log(float(eps))

    def alpha_t(self, t: torch.Tensor) -> torch.Tensor:
        return torch.exp(-self.lam * t)

    def alpha_prime_t(self, t: torch.Tensor) -> torch.Tensor:
        return -self.lam * torch.exp(-self.lam * t)


__all__ = ["Exponential"]
