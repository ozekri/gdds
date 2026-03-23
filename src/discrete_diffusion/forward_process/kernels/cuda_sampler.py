"""JIT loader for the custom block sampler extension."""

from __future__ import annotations

import importlib
import os
from pathlib import Path
from typing import Any

import torch
from torch.utils.cpp_extension import CUDA_HOME, load


_EXTENSION = None
_LOAD_ERROR = None


def _extension_name() -> str:
    suffix = "cuda" if torch.cuda.is_available() and CUDA_HOME else "cpu"
    return f"gdds_block_sampler_{suffix}"


def _extension_sources() -> tuple[list[str], bool]:
    root = Path(__file__).resolve().parent / "csrc"
    cpp = str(root / "block_sampler.cpp")
    cu = str(root / "block_sampler_cuda.cu")
    use_cuda = torch.cuda.is_available() and CUDA_HOME is not None and Path(cu).exists()
    sources = [cpp]
    if use_cuda:
        sources.append(cu)
    return sources, use_cuda


def load_block_sampler_extension() -> Any:
    global _EXTENSION, _LOAD_ERROR
    if _EXTENSION is not None:
        return _EXTENSION
    if _LOAD_ERROR is not None:
        raise RuntimeError("block sampler extension failed to load previously") from _LOAD_ERROR

    name = _extension_name()
    try:
        _EXTENSION = importlib.import_module(name)
        return _EXTENSION
    except Exception:
        pass

    sources, use_cuda = _extension_sources()
    extra_cflags = ["-O3"]
    extra_cuda_cflags = ["-O3", "--use_fast_math", "--expt-relaxed-constexpr"]
    if use_cuda:
        extra_cflags.append("-DWITH_CUDA")
    build_directory = os.path.join("/tmp", "gdds_extensions")
    os.makedirs(build_directory, exist_ok=True)

    try:
        _EXTENSION = load(
            name=name,
            sources=sources,
            extra_cflags=extra_cflags,
            extra_cuda_cflags=extra_cuda_cflags if use_cuda else None,
            with_cuda=use_cuda,
            build_directory=build_directory,
            verbose=False,
        )
        return _EXTENSION
    except Exception as exc:
        _LOAD_ERROR = exc
        raise


def sample_block_gumbel_argmax(chunk_logR: torch.Tensor, chunk_exp: torch.Tensor, seed: int):
    ext = load_block_sampler_extension()
    return ext.sample_block_gumbel_argmax(chunk_logR, chunk_exp, int(seed))


def sample_block_gumbel_argmax_indexed(
    unique_logR: torch.Tensor,
    row_index: torch.Tensor,
    chunk_exp: torch.Tensor,
    seed: int,
):
    ext = load_block_sampler_extension()
    return ext.sample_block_gumbel_argmax_indexed(unique_logR, row_index, chunk_exp, int(seed))
