"""FlexMDM schedule primitives and factory."""

from __future__ import annotations

import abc
from typing import Any, Mapping

import torch
from torch import Tensor


def _to_dict(config: Any) -> dict[str, Any]:
  """Convert Hydra configs or plain dicts to a python dict."""
  if config is None:
    return {}
  if isinstance(config, dict):
    return dict(config)

  try:
    from omegaconf import OmegaConf  # type: ignore
  except ImportError as exc:  # pragma: no cover - hydra is part of runtime deps
    raise TypeError(
      "Flex schedule configuration must be a dict when OmegaConf is unavailable."
    ) from exc

  return OmegaConf.to_container(config, resolve=True)  # type: ignore[arg-type]


class FlexSchedule(abc.ABC):
  """Minimal interface matching FlexMDM's schedule objects.

  Note:
      Flex noise schedules go from 0 to 1 (increasing noise/masking), while
      standard schedules go from 1 to 0 (decreasing signal). These should be
      unified in the future.
  """

  @abc.abstractmethod
  def at(self, t: Tensor) -> Tensor:
    raise NotImplementedError

  @abc.abstractmethod
  def derivative_at(self, t: Tensor) -> Tensor:
    raise NotImplementedError

  @abc.abstractmethod
  def inv(self, alpha: Tensor) -> Tensor:
    raise NotImplementedError

  def rate_scale_factor(self, t: Tensor) -> Tensor:
    denom = (1 - self.at(t)).clamp_min(1e-6)
    return self.derivative_at(t) / denom

  def sample(self, shape: tuple[int, ...], device: torch.device) -> Tensor:
    uniform = torch.rand(shape, device=device)
    return self.inv(uniform)

  def sample_truncated(
    self, threshold: Tensor, shape: tuple[int, ...], device: torch.device
  ) -> Tensor:
    uniform = torch.rand(shape, device=device)
    threshold_alpha = self.at(threshold)
    return self.inv(uniform * (1 - threshold_alpha) + threshold_alpha)


class LinearSchedule(FlexSchedule):
  def at(self, t: Tensor) -> Tensor:
    return t

  def derivative_at(self, t: Tensor) -> Tensor:
    return torch.ones_like(t, device=t.device)

  def inv(self, alpha: Tensor) -> Tensor:
    return alpha


class GeometricSchedule(FlexSchedule):
  def __init__(self, min_val: float, max_val: float):
    self.min = float(min_val)
    self.max = float(max_val)

  def _broadcast(self, value: float, ref: Tensor) -> Tensor:
    return torch.as_tensor(value, device=ref.device, dtype=ref.dtype)

  def at(self, t: Tensor) -> Tensor:
    min_val = self._broadcast(self.min, t)
    max_val = self._broadcast(self.max, t)
    return torch.exp(-(min_val ** (1 - t)) * max_val**t)

  def derivative_at(self, t: Tensor) -> Tensor:
    min_val = self._broadcast(self.min, t)
    max_val = self._broadcast(self.max, t)
    return (
      self.at(t) * min_val ** (1 - t) * max_val**t * (min_val.log() - max_val.log())
    )

  def inv(self, alpha: Tensor) -> Tensor:
    log_min = torch.log(self._broadcast(self.min, alpha))
    log_max = torch.log(self._broadcast(self.max, alpha))
    return (torch.log(-torch.log(alpha)) - log_min) / (log_max - log_min)


class SinSchedule(FlexSchedule):
  def at(self, t: Tensor) -> Tensor:
    return torch.sin(torch.pi / 2 * t)

  def derivative_at(self, t: Tensor) -> Tensor:
    return (torch.pi / 2) * torch.cos(torch.pi / 2 * t)

  def inv(self, alpha: Tensor) -> Tensor:
    return (2 / torch.pi) * torch.asin(alpha.clamp(min=0.0, max=1.0))


class CosineSchedule(FlexSchedule):
  def at(self, t: Tensor) -> Tensor:
    return 1 - torch.cos(torch.pi / 2 * t)

  def derivative_at(self, t: Tensor) -> Tensor:
    return (torch.pi / 2) * torch.sin(torch.pi / 2 * t)

  def rate_scale_factor(self, t: Tensor) -> Tensor:
    return (torch.pi / 2) * torch.tan(torch.pi / 2 * t)

  def inv(self, alpha: Tensor) -> Tensor:
    clipped = alpha.clamp(min=0.0, max=1.0)
    return (2 / torch.pi) * torch.arccos(1 - clipped)


class PolynomialSchedule(FlexSchedule):
  def __init__(self, exp: float):
    self.exp = float(exp)

  def at(self, t: Tensor) -> Tensor:
    return t**self.exp

  def derivative_at(self, t: Tensor) -> Tensor:
    return self.exp * t ** (self.exp - 1)

  def inv(self, alpha: Tensor) -> Tensor:
    return alpha ** (1 / self.exp)


def build_flex_schedule(config: Mapping[str, Any] | None) -> FlexSchedule:
  """Instantiate a Flex-style schedule from a Hydra config snippet."""
  cfg = _to_dict(config)
  schedule_type = cfg.get("type", "linear").lower()

  if schedule_type == "linear":
    return LinearSchedule()
  if schedule_type == "cosine":
    return CosineSchedule()
  if schedule_type == "sin":
    return SinSchedule()
  if schedule_type == "polynomial":
    if "exp" not in cfg:
      raise ValueError("Polynomial schedule requires 'exp'.")
    return PolynomialSchedule(exp=float(cfg["exp"]))
  if schedule_type == "geometric":
    missing = [k for k in ("min", "max") if k not in cfg]
    if missing:
      raise ValueError(f"Geometric schedule missing keys: {missing}")
    return GeometricSchedule(min_val=float(cfg["min"]), max_val=float(cfg["max"]))

  raise ValueError(f"Unsupported Flex schedule type: {schedule_type}")


__all__ = [
  "FlexSchedule",
  "LinearSchedule",
  "CosineSchedule",
  "SinSchedule",
  "PolynomialSchedule",
  "GeometricSchedule",
  "build_flex_schedule",
]

