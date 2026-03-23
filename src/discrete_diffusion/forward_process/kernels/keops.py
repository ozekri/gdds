"""KeOps (Lazy) implementation of the SIK Kernel."""

from __future__ import annotations
from typing import Literal
from itertools import count
import time
import torch
from .base import SIKKernel
from .cuda_sampler import (
    load_block_sampler_extension,
    sample_block_gumbel_argmax as sample_block_gumbel_argmax_ext,
    sample_block_gumbel_argmax_indexed as sample_block_gumbel_argmax_indexed_ext,
)

try:
    import triton
    import triton.language as tl
    _TRITON_AVAILABLE = True
except ImportError:
    triton = None  # type: ignore[assignment]
    tl = None  # type: ignore[assignment]
    _TRITON_AVAILABLE = False


def _sample_block_gumbel_argmax(chunk_logR: torch.Tensor, chunk_exp: torch.Tensor):
    """Fused block sampler: exact Gumbel-max over one [B, V] score block."""
    scores = torch.empty_like(chunk_logR).exponential_().log_().neg_()
    scores.addcmul_(chunk_exp.unsqueeze(1), chunk_logR)
    return scores.max(dim=1)


_CUDA_SAMPLER_SEED_COUNTER = count()


def _sample_block_gumbel_argmax_cuda_ext(chunk_logR: torch.Tensor, chunk_exp: torch.Tensor):
    seed = next(_CUDA_SAMPLER_SEED_COUNTER) & 0x7FFFFFFF
    return sample_block_gumbel_argmax_ext(chunk_logR, chunk_exp, seed)


def _sample_block_gumbel_argmax_cuda_ext_indexed(
    unique_logR: torch.Tensor,
    row_index: torch.Tensor,
    chunk_exp: torch.Tensor,
):
    seed = next(_CUDA_SAMPLER_SEED_COUNTER) & 0x7FFFFFFF
    return sample_block_gumbel_argmax_indexed_ext(unique_logR, row_index, chunk_exp, seed)


if _TRITON_AVAILABLE:
    _TRITON_SEED_COUNTER = count()

    @triton.jit(
        do_not_specialize=[
            "stride_logr0",
            "stride_logr1",
            "num_rows",
            "vocab_size",
            "seed",
        ]
    )
    def _sample_block_gumbel_argmax_kernel(
        logr_ptr,
        exp_ptr,
        out_scores_ptr,
        out_indices_ptr,
        stride_logr0,
        stride_logr1,
        num_rows,
        vocab_size,
        seed,
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
    ):
        pid_m = tl.program_id(0)
        row_offsets = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        row_mask = row_offsets < num_rows

        exponents = tl.load(exp_ptr + row_offsets, mask=row_mask, other=0.0).to(tl.float32)
        best_scores = tl.full((BLOCK_M,), -float("inf"), tl.float32)
        best_indices = tl.zeros((BLOCK_M,), tl.int32)

        for col_start in range(0, 4096, BLOCK_N):
            col_offsets = col_start + tl.arange(0, BLOCK_N)
            col_mask = col_offsets < vocab_size
            ptrs = logr_ptr + row_offsets[:, None] * stride_logr0 + col_offsets[None, :] * stride_logr1
            mask = row_mask[:, None] & col_mask[None, :]
            log_r = tl.load(ptrs, mask=mask, other=-float("inf")).to(tl.float32)

            rand_offsets = row_offsets[:, None] * vocab_size + col_offsets[None, :]
            uniform = tl.rand(seed, rand_offsets)
            uniform = tl.maximum(uniform, 1.0e-7)
            gumbel = -tl.log(-tl.log(uniform))
            scores = tl.where(mask, exponents[:, None] * log_r + gumbel, -float("inf"))

            tile_best = tl.max(scores, axis=1)
            tile_arg = tl.argmax(scores, axis=1) + col_start
            update = tile_best > best_scores
            best_scores = tl.where(update, tile_best, best_scores)
            best_indices = tl.where(update, tile_arg, best_indices)

        tl.store(out_scores_ptr + row_offsets, best_scores, mask=row_mask)
        tl.store(out_indices_ptr + row_offsets, best_indices, mask=row_mask)


def _sample_block_gumbel_argmax_triton(chunk_logR: torch.Tensor, chunk_exp: torch.Tensor):
    """Triton block sampler computing the final rowwise max/argmax for one vocab block."""
    if not _TRITON_AVAILABLE:
        raise RuntimeError("Triton is not available")
    if chunk_logR.device.type != "cuda":
        raise RuntimeError("Triton sampler requires CUDA tensors")
    if chunk_logR.stride(-1) != 1:
        chunk_logR = chunk_logR.contiguous()
    if chunk_exp.stride(-1) != 1:
        chunk_exp = chunk_exp.contiguous()

    num_rows, vocab_size = chunk_logR.shape
    if num_rows == 0 or vocab_size == 0:
        empty_scores = torch.empty((num_rows,), device=chunk_logR.device, dtype=torch.float32)
        empty_indices = torch.empty((num_rows,), device=chunk_logR.device, dtype=torch.long)
        return empty_scores, empty_indices
    # For tiny row counts, launch overhead dominates and the torch path is faster/stabler.
    if num_rows < 256 or vocab_size > 4096:
        return _sample_block_gumbel_argmax(chunk_logR, chunk_exp)

    out_scores = torch.empty((num_rows,), device=chunk_logR.device, dtype=torch.float32)
    out_indices = torch.empty((num_rows,), device=chunk_logR.device, dtype=torch.int32)
    # Keep the launch seed on the host to avoid an implicit device sync from `.item()`.
    seed = next(_TRITON_SEED_COUNTER) & 0x7FFFFFFF
    if vocab_size <= 1024:
        block_n = 1024
        block_m = 8
        num_warps = 4
    elif vocab_size <= 2048:
        block_n = 1024
        block_m = 8
        num_warps = 8
    else:
        block_n = 1024
        block_m = 8
        num_warps = 8

    grid = (triton.cdiv(num_rows, block_m),)
    _sample_block_gumbel_argmax_kernel[grid](
        chunk_logR,
        chunk_exp,
        out_scores,
        out_indices,
        chunk_logR.stride(0),
        chunk_logR.stride(1),
        num_rows,
        vocab_size,
        seed,
        BLOCK_M=block_m,
        BLOCK_N=block_n,
        num_warps=num_warps,
        num_stages=4,
    )

    return out_scores, out_indices.to(torch.long)

class KeOpsKernel(SIKKernel):
    """SIK Kernel using KeOps for lazy GPU-accelerated computation."""
    
    def __init__(
        self,
        embeddings: torch.Tensor,
        epsilon: float = 0.01,
        gamma: float = 0.0,
        metric: Literal["gaussian", "cosine"] = "gaussian",
        variable_bandwidth: bool = True,
        k_neighbors: int = 7,
        pos_chunk_size: int = 2048,
        vocab_block_size: int = 2048,
        unique_token_chunk_size: int = 4096,
        use_bf16: bool = False,
        verbose: bool = False,
        use_compiled_sampler: bool = False,
        use_triton_sampler: bool = False,
        use_cuda_sampler: bool = False,
    ):
        super().__init__(embeddings, epsilon, gamma, metric, variable_bandwidth, k_neighbors)
        
        try:
            from pykeops.torch import LazyTensor
            self.LazyTensor = LazyTensor
        except ImportError:
            raise ImportError("pykeops is required for KeOpsKernel. Install it with: pip install pykeops")

        if metric == "cosine":
            self.embeddings = self.embeddings / self.embeddings.norm(dim=1, keepdim=True).clamp(min=1e-10)
        self.pos_chunk_size = pos_chunk_size
        self.vocab_block_size = vocab_block_size
        self.unique_token_chunk_size = unique_token_chunk_size
        self.use_bf16 = bool(use_bf16)
        self.verbose = bool(verbose)
        self.use_compiled_sampler = bool(use_compiled_sampler)
        self.use_triton_sampler = bool(use_triton_sampler)
        self.use_cuda_sampler = bool(use_cuda_sampler)
        self._emb_sqnorm = (self.embeddings * self.embeddings).sum(dim=1) if metric == "gaussian" else None
        self._sample_dtype = torch.bfloat16 if self.use_bf16 and self.embeddings.device.type == "cuda" else self.embeddings.dtype
        self._sample_embeddings = self.embeddings.to(self._sample_dtype)
        self._sample_sigma = None
        self._sample_inv_sigma = None
        self._sample_emb_over_sigma = None
        self._sample_sqnorm_over_sigma = None
        self._sample_emb_sqnorm = self._emb_sqnorm.to(self._sample_dtype) if self._emb_sqnorm is not None else None
        self._vocab_ids = torch.arange(self.m, device=self.embeddings.device, dtype=torch.long)

        self._sigma = None
        if variable_bandwidth:
            self._sigma = self._compute_local_bandwidths()
            self._sample_sigma = self._sigma.to(self._sample_dtype)
            self._sample_inv_sigma = self._sample_sigma.reciprocal()
            if self.metric == "gaussian":
                self._sample_emb_over_sigma = self._sample_embeddings * self._sample_inv_sigma.unsqueeze(1)
                self._sample_sqnorm_over_sigma = self._sample_emb_sqnorm * self._sample_inv_sigma

        if gamma > 0:
            # Pre-compute normalization factors (d_i) only when they are needed.
            print(f"[KeOps] Pre-computing degrees...")
            K = self._build_lazy_kernel()
            self.d = K.sum(dim=1).squeeze()
            self.n_i = self.d.pow(gamma)
            self._log_n_i = self.n_i.clamp(min=1e-10).log()
            self._sample_log_n_i = self._log_n_i.to(self._sample_dtype)
        else:
            self.d = None
            self.n_i = None
            self._log_n_i = None
            self._sample_log_n_i = None

        self._block_sampler = _sample_block_gumbel_argmax
        self._indexed_block_sampler = None
        self._triton_warmed_up = False
        self._cuda_sampler_warmed_up = False
        if self.use_cuda_sampler:
            load_block_sampler_extension()
            self._block_sampler = _sample_block_gumbel_argmax_cuda_ext
            self._indexed_block_sampler = _sample_block_gumbel_argmax_cuda_ext_indexed
        elif self.use_triton_sampler:
            if not _TRITON_AVAILABLE:
                raise ImportError("use_triton_sampler=True but Triton is not installed in this environment")
            self._block_sampler = _sample_block_gumbel_argmax_triton
        elif self.use_compiled_sampler and hasattr(torch, "compile"):
            self._block_sampler = torch.compile(  # type: ignore[assignment]
                _sample_block_gumbel_argmax,
                dynamic=True,
                mode="max-autotune-no-cudagraphs",
            )

    def _maybe_warmup_triton_sampler(self, math_dtype: torch.dtype) -> None:
        if not self.use_triton_sampler or self._triton_warmed_up or self.device.type != "cuda":
            return
        # Warm up representative specializations once so compile cost stays out of timings.
        for warm_rows, warm_cols in (
            (max(256, min(1024, self.pos_chunk_size)), min(1024, self.vocab_block_size)),
            (max(256, min(1024, self.pos_chunk_size)), min(2048, self.vocab_block_size)),
            (max(256, min(1024, self.pos_chunk_size)), min(4096, self.vocab_block_size)),
        ):
            if warm_rows <= 0 or warm_cols <= 0:
                continue
            warm_logr = torch.zeros((warm_rows, warm_cols), device=self.device, dtype=math_dtype)
            warm_exp = torch.ones((warm_rows,), device=self.device, dtype=math_dtype)
            _ = self._block_sampler(warm_logr, warm_exp)
        torch.cuda.synchronize()
        self._triton_warmed_up = True

    def _maybe_warmup_cuda_sampler(self, math_dtype: torch.dtype) -> None:
        if not self.use_cuda_sampler or self._cuda_sampler_warmed_up or self.device.type != "cuda":
            return
        load_block_sampler_extension()
        for warm_rows, warm_cols in (
            (max(256, min(1024, self.pos_chunk_size)), min(1024, self.vocab_block_size)),
            (max(256, min(1024, self.pos_chunk_size)), min(2048, self.vocab_block_size)),
            (max(256, min(1024, self.pos_chunk_size)), min(4096, self.vocab_block_size)),
        ):
            if warm_rows <= 0 or warm_cols <= 0:
                continue
            warm_logr = torch.zeros((warm_rows, warm_cols), device=self.device, dtype=math_dtype)
            warm_exp = torch.ones((warm_rows,), device=self.device, dtype=math_dtype)
            _ = self._block_sampler(warm_logr, warm_exp)
        torch.cuda.synchronize()
        self._cuda_sampler_warmed_up = True

    def _compute_local_bandwidths(self) -> torch.Tensor:
        with torch.no_grad():
            x_i = self.LazyTensor(self.embeddings.unsqueeze(1))
            x_j = self.LazyTensor(self.embeddings.unsqueeze(0))
            if self.metric == "gaussian":
                D_ij = x_i.sqdist(x_j).sqrt()
            else:
                cos_sim = (x_i * x_j).sum(dim=2)
                D_ij = 1.0 - cos_sim
            
            kth_dists = D_ij.Kmin(self.k_neighbors + 1, dim=1)[:, self.k_neighbors]
            return kth_dists.squeeze().clamp(min=1e-10)

    def _effective_unique_token_chunk_size(self) -> int:
        if not self.use_cuda_sampler or self.device.type != "cuda":
            return self.unique_token_chunk_size
        # Bigger unique-token blocks reduce Python/launch overhead and give GEMMs
        # a more efficient shape on H100. Keep the heuristic conservative enough
        # to avoid blowing up memory in the exact dense path.
        target = 8192 if self._sample_dtype == torch.bfloat16 else 6144
        return max(self.unique_token_chunk_size, target)

    def _build_lazy_kernel(self):
        """Build symbolic K_eps using KeOps."""
        x_i = self.LazyTensor(self.embeddings.unsqueeze(1))
        x_j = self.LazyTensor(self.embeddings.unsqueeze(0))
        
        if self.metric == "gaussian":
            dist = x_i.sqdist(x_j)
        else:
            cos_sim = (x_i * x_j).sum(dim=2)
            dist = 1.0 - cos_sim

        if self.variable_bandwidth:
            s_i = self.LazyTensor(self._sigma.unsqueeze(1).unsqueeze(2))
            s_j = self.LazyTensor(self._sigma.unsqueeze(0).unsqueeze(2))
            bw = s_i * s_j
            return (-dist / (self.epsilon * bw)).exp()
        return (-dist / self.epsilon).exp()

    def sample_neighbors(self, token_ids: torch.Tensor, exponent: torch.Tensor) -> torch.Tensor:
        """Sample next tokens without materializing the full [N, V] logits tensor.

        We use an exact streaming Gumbel-max over vocabulary blocks. This keeps memory
        bounded while remaining mathematically equivalent to sampling from the full
        categorical defined by the KeOps kernel.
        """
        device = token_ids.device
        N = token_ids.shape[0]
        if N == 0:
            return token_ids

        all_embeddings = self._sample_embeddings
        all_log_n = self._sample_log_n_i
        all_sigma = self._sample_sigma if self.variable_bandwidth else None
        out = torch.empty(N, dtype=torch.long, device=device)
        math_dtype = self._sample_dtype
        self._maybe_warmup_cuda_sampler(math_dtype)
        self._maybe_warmup_triton_sampler(math_dtype)
        t_total_start = time.perf_counter() if self.verbose else 0.0
        stats_unique = 0.0
        stats_dist = 0.0
        stats_kernel = 0.0
        stats_gumbel = 0.0
        stats_argmax = 0.0

        # Reuse deterministic source-token work across repeated tokens.
        t0 = time.perf_counter() if self.verbose else 0.0
        unique_tokens, inverse_indices = torch.unique(token_ids, sorted=True, return_inverse=True)
        sort_perm = torch.argsort(inverse_indices)
        sorted_inverse = inverse_indices[sort_perm]
        if self.verbose:
            if device.type == "cuda":
                torch.cuda.synchronize()
            stats_unique += time.perf_counter() - t0

        effective_unique_chunk_size = self._effective_unique_token_chunk_size()
        for uniq_start in range(0, unique_tokens.shape[0], effective_unique_chunk_size):
            uniq_end = min(uniq_start + effective_unique_chunk_size, unique_tokens.shape[0])
            bounds = torch.searchsorted(
                sorted_inverse,
                torch.tensor([uniq_start, uniq_end], device=device, dtype=sorted_inverse.dtype),
            )
            pos_block_start = int(bounds[0].item())
            pos_block_end = int(bounds[1].item())
            if pos_block_start == pos_block_end:
                continue

            block_positions = sort_perm[pos_block_start:pos_block_end]
            block_local_inverse = sorted_inverse[pos_block_start:pos_block_end] - uniq_start
            block_exponent = exponent[block_positions].float()

            uniq_tokens_block = unique_tokens[uniq_start:uniq_end]
            src_emb = all_embeddings[uniq_tokens_block]
            log_n_src = all_log_n[uniq_tokens_block] if all_log_n is not None else None
            sigma_src = all_sigma[uniq_tokens_block] if all_sigma is not None else None
            src_sqnorm = self._sample_emb_sqnorm[uniq_tokens_block] if self._sample_emb_sqnorm is not None else None
            src_inv_sigma = self._sample_inv_sigma[uniq_tokens_block] if self._sample_inv_sigma is not None else None
            src_emb_over_sigma = (
                self._sample_emb_over_sigma[uniq_tokens_block]
                if self._sample_emb_over_sigma is not None
                else None
            )
            src_sqnorm_over_sigma = (
                self._sample_sqnorm_over_sigma[uniq_tokens_block]
                if self._sample_sqnorm_over_sigma is not None
                else None
            )

            best_score = torch.full((block_positions.shape[0],), float("-inf"), device=device)
            best_idx = torch.zeros((block_positions.shape[0],), dtype=torch.long, device=device)

            for vocab_start in range(0, self.m, self.vocab_block_size):
                vocab_end = min(vocab_start + self.vocab_block_size, self.m)
                tgt_emb = all_embeddings[vocab_start:vocab_end]

                t1 = time.perf_counter() if self.verbose else 0.0
                if self.metric == "gaussian":
                    if sigma_src is not None:
                        tgt_inv_sigma = self._sample_inv_sigma[vocab_start:vocab_end]
                        tgt_emb_over_sigma = self._sample_emb_over_sigma[vocab_start:vocab_end]
                        mat = src_emb_over_sigma @ tgt_emb_over_sigma.t()
                    else:
                        mat = src_emb @ tgt_emb.t()
                else:
                    mat = torch.mm(src_emb, tgt_emb.t())
                if self.verbose:
                    if device.type == "cuda":
                        torch.cuda.synchronize()
                    stats_dist += time.perf_counter() - t1

                t2 = time.perf_counter() if self.verbose else 0.0
                if self.metric == "gaussian":
                    if sigma_src is not None:
                        tgt_inv_sigma = self._sample_inv_sigma[vocab_start:vocab_end]
                        tgt_sqnorm_over_sigma = self._sample_sqnorm_over_sigma[vocab_start:vocab_end]
                        logR_unique = mat.mul(2.0 / self.epsilon)
                        logR_unique.addcmul_(
                            src_sqnorm_over_sigma.unsqueeze(1),
                            tgt_inv_sigma.unsqueeze(0),
                            value=-1.0 / self.epsilon,
                        )
                        logR_unique.addcmul_(
                            src_inv_sigma.unsqueeze(1),
                            tgt_sqnorm_over_sigma.unsqueeze(0),
                            value=-1.0 / self.epsilon,
                        )
                    else:
                        tgt_sqnorm = self._sample_emb_sqnorm[vocab_start:vocab_end]
                        logR_unique = mat.mul(2.0 / self.epsilon)
                        logR_unique.add_(src_sqnorm.unsqueeze(1), alpha=-1.0 / self.epsilon)
                        logR_unique.add_(tgt_sqnorm.unsqueeze(0), alpha=-1.0 / self.epsilon)
                else:
                    logR_unique = mat.add(-1.0)
                    if sigma_src is not None:
                        tgt_inv_sigma = self._sample_inv_sigma[vocab_start:vocab_end]
                        logR_unique.mul_(src_inv_sigma.unsqueeze(1))
                        logR_unique.mul_(tgt_inv_sigma.unsqueeze(0))
                    logR_unique.div_(self.epsilon)

                if log_n_src is not None:
                    logR_unique.sub_(log_n_src.unsqueeze(1))
                    logR_unique.sub_(all_log_n[vocab_start:vocab_end].unsqueeze(0))

                # Avoid building a full [U, V] boolean mask just to suppress self-transitions.
                in_block = (uniq_tokens_block >= vocab_start) & (uniq_tokens_block < vocab_end)
                if in_block.any():
                    row_ids = in_block.nonzero(as_tuple=False).squeeze(1)
                    col_ids = uniq_tokens_block[row_ids] - vocab_start
                    logR_unique[row_ids, col_ids] = float("-inf")
                if self.verbose:
                    if device.type == "cuda":
                        torch.cuda.synchronize()
                    stats_kernel += time.perf_counter() - t2

                if self._indexed_block_sampler is not None:
                    chunk_exp = block_exponent.to(math_dtype)

                    t3 = time.perf_counter() if self.verbose else 0.0
                    block_score, block_argmax = self._indexed_block_sampler(
                        logR_unique,
                        block_local_inverse,
                        chunk_exp,
                    )
                    if self.verbose:
                        if device.type == "cuda":
                            torch.cuda.synchronize()
                        stats_gumbel += time.perf_counter() - t3
                    t4 = time.perf_counter() if self.verbose else 0.0
                    block_argmax = block_argmax + vocab_start
                    if self.verbose:
                        if device.type == "cuda":
                            torch.cuda.synchronize()
                        stats_argmax += time.perf_counter() - t4

                    update_mask = block_score > best_score
                    best_score = torch.where(update_mask, block_score, best_score)
                    best_idx = torch.where(update_mask, block_argmax, best_idx)
                else:
                    for pos_start in range(0, block_positions.shape[0], self.pos_chunk_size):
                        pos_end = min(pos_start + self.pos_chunk_size, block_positions.shape[0])
                        chunk_local_inverse = block_local_inverse[pos_start:pos_end]
                        chunk_exp = block_exponent[pos_start:pos_end].to(math_dtype)

                        t3 = time.perf_counter() if self.verbose else 0.0
                        chunk_logR = logR_unique[chunk_local_inverse]
                        block_score, block_argmax = self._block_sampler(chunk_logR, chunk_exp)
                        if self.verbose:
                            if device.type == "cuda":
                                torch.cuda.synchronize()
                            stats_gumbel += time.perf_counter() - t3
                        t4 = time.perf_counter() if self.verbose else 0.0
                        block_argmax = block_argmax + vocab_start
                        if self.verbose:
                            if device.type == "cuda":
                                torch.cuda.synchronize()
                            stats_argmax += time.perf_counter() - t4

                        view = slice(pos_start, pos_end)
                        update_mask = block_score > best_score[view]
                        best_score[view] = torch.where(update_mask, block_score, best_score[view])
                        best_idx[view] = torch.where(update_mask, block_argmax, best_idx[view])

            out[block_positions] = best_idx

        if self.verbose:
            if device.type == "cuda":
                torch.cuda.synchronize()
            total = time.perf_counter() - t_total_start
            print(
                "[KeOps sample_neighbors] "
                f"N={N} unique={unique_tokens.shape[0]} total={total*1000:.2f}ms "
                f"(unique={stats_unique*1000:.2f}ms, dist={stats_dist*1000:.2f}ms, "
                f"kernel={stats_kernel*1000:.2f}ms, gumbel={stats_gumbel*1000:.2f}ms, "
                f"argmax={stats_argmax*1000:.2f}ms, other={(total-stats_unique-stats_dist-stats_kernel-stats_gumbel-stats_argmax)*1000:.2f}ms)"
            )

        return out
