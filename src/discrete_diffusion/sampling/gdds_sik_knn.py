"""GDDS-SIK sampler for KNN forward processes."""

from __future__ import annotations

import bisect
import math
from typing import Any, Callable, Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F

from .base import Sampler

DEFAULT_TRANSPOSE = False


class BufferManager:

    def __init__(self):
        self._buffers: Dict[str, torch.Tensor] = {}
        self._buffer_keys: Dict[str, tuple] = {}

    def get_buffer(
        self,
        name: str,
        shape: tuple,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        key = (name, device, dtype, shape)
        if name in self._buffers:
            existing = self._buffers[name]
            if (self._buffer_keys.get(name) == key and
                existing.shape == shape and
                existing.device == device and
                existing.dtype == dtype):
                return existing
        self._buffers[name] = torch.empty(shape, device=device, dtype=dtype)
        self._buffer_keys[name] = key
        return self._buffers[name]

    def clear(self):
        self._buffers.clear()
        self._buffer_keys.clear()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


class SparseCSRMatmulCache:
    __slots__ = ('_incoming_ptr', '_incoming_src', '_incoming_slot', '_V', '_device', '_dtype',
                 '_cache_by_weights_id', '_max_cache_entries')

    def __init__(
        self,
        incoming_ptr: torch.Tensor,
        incoming_src: torch.Tensor,
        incoming_slot: torch.Tensor,
        V: int,
        device: torch.device,
        dtype: torch.dtype = torch.float32,
        max_cache_entries: int = 8,
    ):
        self._incoming_ptr = incoming_ptr
        self._incoming_src = incoming_src
        self._incoming_slot = incoming_slot
        self._V = V
        self._device = device
        self._dtype = dtype
        self._cache_by_weights_id: Dict[int, Tuple[torch.Tensor, torch.Tensor]] = {}
        self._max_cache_entries = max(1, int(max_cache_entries))

    def get_csr(self, weights: torch.Tensor, transpose: bool) -> torch.Tensor:
        w_id = id(weights)
        mats = self._cache_by_weights_id.get(w_id)
        if mats is None:
            mats = self._rebuild(weights)
            if len(self._cache_by_weights_id) >= self._max_cache_entries:
                oldest_key = next(iter(self._cache_by_weights_id))
                del self._cache_by_weights_id[oldest_key]
            self._cache_by_weights_id[w_id] = mats
        return mats[1] if transpose else mats[0]

    def _rebuild(self, weights: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        if torch.isnan(weights).any() or torch.isinf(weights).any():
            raise ValueError("weights contain NaN or inf; cannot build sparse CSR.")
        values = weights[self._incoming_src, self._incoming_slot].contiguous().to(
            device=self._device, dtype=self._dtype
        )
        A_in = torch.sparse_csr_tensor(
            self._incoming_ptr,
            self._incoming_src,
            values,
            size=(self._V, self._V),
            device=self._device,
            dtype=self._dtype,
        )
        A_in_T = A_in.t()  # F_sem^T
        return A_in, A_in_T

    def clear(self) -> None:
        self._cache_by_weights_id.clear()

def apply_U_exclusive_out(
    X: torch.Tensor,               # [V, B]
    out: torch.Tensor,             # [V, B]
    total_buf: torch.Tensor,       # [1, B]
) -> None:
    V, B = X.shape
    if V <= 1:
        out.zero_()
        return
    torch.sum(X, dim=0, keepdim=True, out=total_buf)   # [1,B]
    out.copy_(X)
    out.mul_(-1.0)
    out.add_(total_buf)                                # broadcast add
    out.div_(V - 1)

def apply_F_tilde_cached_out(
    X: torch.Tensor,                # [V, B]
    out: torch.Tensor,              # [V, B]
    weights: torch.Tensor,          # [V, k]
    lam: float | torch.Tensor,
    incoming_ptr: torch.Tensor,     # [V+1] CSR row pointers for INCOMING edges
    incoming_src: torch.Tensor,     # [E] source tokens for incoming edges
    incoming_slot: torch.Tensor,    # [E] slot positions for incoming edges
    sem_buf: torch.Tensor,          # [V, B]
    uni_buf: torch.Tensor,          # [V, B]
    total_buf: torch.Tensor,        # [1, B]
    *,
    transpose: bool = DEFAULT_TRANSPOSE,
    csr_cache: Optional[SparseCSRMatmulCache] = None,
) -> None:
    lam_f = float(lam) if isinstance(lam, torch.Tensor) else lam
    V = X.shape[0]
    if lam_f >= 1.0 - 1e-12:
        apply_U_exclusive_out(X, uni_buf, total_buf)
        out.copy_(uni_buf)
        return
    if lam_f <= 1e-12:
        if csr_cache is not None:
            M = csr_cache.get_csr(weights, transpose=transpose)
        else:
            values = weights[incoming_src, incoming_slot].contiguous()
            A_in = torch.sparse_csr_tensor(
                incoming_ptr, incoming_src, values,
                size=(V, V), device=X.device, dtype=X.dtype,
            )
            M = A_in.t() if transpose else A_in
        tmp = torch.sparse.mm(M, X)
        out.copy_(tmp)
        return
    if csr_cache is not None:
        M = csr_cache.get_csr(weights, transpose=transpose)
    else:
        values = weights[incoming_src, incoming_slot].contiguous()
        A_in = torch.sparse_csr_tensor(
            incoming_ptr, incoming_src, values,
            size=(V, V), device=X.device, dtype=X.dtype,
        )
        M = A_in.t() if transpose else A_in
    tmp = torch.sparse.mm(M, X)
    sem_buf.copy_(tmp)
    apply_U_exclusive_out(X, uni_buf, total_buf)
    out.copy_(sem_buf)
    out.mul_(1.0 - lam_f)
    out.add_(uni_buf, alpha=lam_f)

def compute_truncation(
    mu: float,
    eps_abs: float,
    eps_rel: float = 1e-6,
    N_max: int = 256,
) -> int:
    if mu <= 0:
        return 0
    p_n = math.exp(-mu)
    cdf = p_n
    coeff_sum = p_n
    N = 0
    while N < N_max:
        tail_ok = (1.0 - cdf) <= eps_abs
        next_coeff = p_n * mu / (N + 1) if (N + 1) <= N_max else 0.0
        rel_ok = True if eps_rel <= 0.0 else (coeff_sum <= 0 or next_coeff <= eps_rel * coeff_sum)
        if tail_ok and rel_ok:
            break
        N += 1
        if N >= N_max:
            break
        p_n = p_n * mu / N
        cdf += p_n
        coeff_sum += p_n
    return min(N, N_max)

def uniformization_apply_block_inplace(
    Y: torch.Tensor,  # [V, B] - modified in-place
    mu_b: float,
    weights_b: torch.Tensor,  # [V, k] - precomputed
    lam_b: float | torch.Tensor,
    incoming_ptr: torch.Tensor,  # [V+1] CSR row pointers for INCOMING edges
    incoming_src: torch.Tensor,  # [E] source tokens for incoming edges
    incoming_slot: torch.Tensor,  # [E] slot positions for incoming edges
    Z_buffer: torch.Tensor,  # [V, B] - current power term
    Z_next: torch.Tensor,  # [V, B] - next power term
    sem_buf: torch.Tensor,  # [V, B]
    uni_buf: torch.Tensor,  # [V, B]
    total_buf: torch.Tensor,  # [1, B]
    N: int,  # Precomputed Poisson truncation level
    *,
    transpose: bool = DEFAULT_TRANSPOSE,
    csr_cache: Optional[SparseCSRMatmulCache] = None,
) -> None:
    V, B = Y.shape
    Z_buffer.copy_(Y)
    Z_next.zero_()
    exp_minus_mu = math.exp(-mu_b)
    a_n = exp_minus_mu
    cdf = a_n  # cumulative mass Σ_{n=0}^k a_n (stable recurrence)
    Y.mul_(a_n)
    for n in range(1, N + 1):
        apply_F_tilde_cached_out(
            Z_buffer, Z_next,
            weights_b, lam_b, incoming_ptr, incoming_src, incoming_slot,
            sem_buf, uni_buf, total_buf,
            transpose=transpose,
            csr_cache=csr_cache,
        )
        Z_buffer, Z_next = Z_next, Z_buffer
        a_n = a_n * mu_b / n
        cdf += a_n
        Y.add_(Z_buffer, alpha=a_n)
    tail = max(0.0, 1.0 - cdf)  # clamp tiny negatives from fp error
    Y.add_(Z_buffer, alpha=tail)

def compute_step_bayes_log_factors_uniformized(
    unique_u: torch.Tensor,
    inv_u: torch.Tensor,
    mu: float,
    N_step: int,
    weights_bar: torch.Tensor,
    lambda_bar: float,
    incoming_ptr: torch.Tensor,
    incoming_src: torch.Tensor,
    incoming_slot: torch.Tensor,
    buffer_manager: BufferManager,
    B: int,
    chunk: int,
    device: torch.device,
    dtype: torch.dtype = torch.float32,
    csr_cache: Optional[SparseCSRMatmulCache] = None,
) -> torch.Tensor:
    U = unique_u.shape[0]
    V = weights_bar.shape[0]
    R = compute_step_bayes_factors_uniformized(
        unique_u, inv_u, mu, N_step,
        weights_bar, lambda_bar,
        incoming_ptr, incoming_src, incoming_slot,
        buffer_manager, B, chunk, device, dtype, csr_cache=csr_cache,
    )
    logR = buffer_manager.get_buffer('logR_bayes', (V, U), device, dtype)
    logR.fill_(-float('inf'))
    mask = R > 0
    if mask.any():
        logR[mask] = torch.log(R[mask])
    return logR

def _debias_x0_posterior_with_xt_likelihood(
    X_theta_chunk: torch.Tensor,  # [B, V, chunk], approx p_theta(x0 | x_t)
    x_t_chunk: torch.Tensor,      # [B, chunk]
    t: float,
    K: int,
    schedule: Any,
    forward_process: Any,
    base_score: torch.Tensor,     # [V, k]
    knn_idx: torch.Tensor,        # [V, k]
    incoming_ptr: torch.Tensor,
    incoming_src: torch.Tensor,
    incoming_slot: torch.Tensor,
    buffer_manager: BufferManager,
    sampling_dtype: torch.dtype,
    poisson_eps_mode: str,
    poisson_eps_coeff: float,
    poisson_eps_rel: float,
    poisson_N_max: int,
    debias_strength: float = 1.0,
    debias_floor: float = 1e-12,
    debias_float64_min_floor: float = 1e-30,
    precomputed_pf_params_shared_full: Optional[List[Tuple[float, torch.Tensor, torch.Tensor, int]]] = None,
    freeze_mu_star: float = 1.0,
    csr_cache: Optional[SparseCSRMatmulCache] = None,
) -> torch.Tensor:
    strength = max(0.0, float(debias_strength))
    if strength <= 1e-12:
        return X_theta_chunk
    B, V, chunk = X_theta_chunk.shape
    device = X_theta_chunk.device
    N_pos = B * chunk
    if N_pos == 0:
        return X_theta_chunk
    x_t_flat = x_t_chunk.reshape(-1).clamp(0, V - 1)
    unique_u, inv_u = torch.unique(x_t_flat, return_inverse=True)
    if unique_u.numel() == 0:
        return X_theta_chunk
    float64_min_floor = float(debias_float64_min_floor)
    if float64_min_floor <= 0.0:
        raise ValueError("debias_float64_min_floor must be > 0.")
    floor = max(float(debias_floor), float64_min_floor if sampling_dtype == torch.float64 else 1e-10)
    log_floor = math.log(floor)

    def _apply_inverse_likelihood(log_q_pos: torch.Tensor) -> torch.Tensor:
        log_q_safe = torch.clamp(log_q_pos, min=log_floor, max=0.0)
        X_flat = X_theta_chunk.permute(0, 2, 1).reshape(N_pos, V)
        x_eps = 1e-30 if sampling_dtype == torch.float64 else 1e-10
        log_x = torch.log(X_flat.clamp(min=x_eps))
        log_x_debiased = log_x - strength * log_q_safe
        X_flat_debiased = F.softmax(log_x_debiased, dim=1)
        return X_flat_debiased.view(B, chunk, V).permute(0, 2, 1).contiguous()
    pf_0t = precomputed_pf_params_shared_full
    if pf_0t is None:
        pf_0t = precompute_pushforward_params(
            float(t), K, schedule, forward_process,
            base_score, knn_idx,
            mu_star=freeze_mu_star,
            device=device,
            poisson_eps_mode=poisson_eps_mode,
            poisson_eps_coeff=poisson_eps_coeff,
            poisson_eps_rel=poisson_eps_rel,
            poisson_N_max=poisson_N_max,
            sampling_dtype=sampling_dtype,
        )
    pf_0t = pf_0t or []
    U = int(unique_u.numel())
    q_rows = buffer_manager.get_buffer('q_rows_xt_debias', (V, U), device, sampling_dtype)
    q_rows.zero_()
    q_rows[unique_u, torch.arange(U, device=device)] = 1.0
    if pf_0t:
        Z_buffer = buffer_manager.get_buffer('Z_xt_debias', (V, U), device, sampling_dtype)
        Z_next = buffer_manager.get_buffer('Z_xt_debias_next', (V, U), device, sampling_dtype)
        sem_buf = buffer_manager.get_buffer('F_sem_xt_debias', (V, U), device, sampling_dtype)
        uni_buf = buffer_manager.get_buffer('U_xt_debias', (V, U), device, sampling_dtype)
        total_buf = buffer_manager.get_buffer('U_total_xt_debias', (1, U), device, sampling_dtype)
        for mu_b, weights_b, lam_b, N_b in reversed(pf_0t):
            uniformization_apply_block_inplace(
                q_rows, float(mu_b), weights_b, lam_b,
                incoming_ptr, incoming_src, incoming_slot,
                Z_buffer, Z_next, sem_buf, uni_buf, total_buf,
                N=int(N_b),
                transpose=True,
                csr_cache=csr_cache,
            )
    log_q_pos = torch.log(q_rows[:, inv_u].t().clamp(min=floor, max=1.0))
    return _apply_inverse_likelihood(log_q_pos)

def sample_chunk_streaming_vocab_with_bayes(
    Y_s_chunk: torch.Tensor,
    logR: torch.Tensor,
    inv_u: torch.Tensor,
    vocab_block_size: int = 50259,
    sampling_dtype: torch.dtype = torch.float32,
    use_argmax: bool = False,
) -> torch.Tensor:
    B, V, chunk = Y_s_chunk.shape
    device = Y_s_chunk.device
    N_pos = B * chunk
    best_val = torch.full((N_pos,), -float("inf"), device=device, dtype=sampling_dtype)
    best_idx = torch.zeros(N_pos, dtype=torch.long, device=device)
    num_vblocks = (V + vocab_block_size - 1) // vocab_block_size
    y_eps = 1e-30 if sampling_dtype == torch.float64 else 1e-10
    for vb_idx in range(num_vblocks):
        v_start = vb_idx * vocab_block_size
        v_end = min(v_start + vocab_block_size, V)
        Y_block = Y_s_chunk[:, v_start:v_end, :].permute(0, 2, 1).reshape(N_pos, v_end - v_start)
        score_block = torch.log(Y_block.clamp(min=y_eps)) + logR[v_start:v_end, :][:, inv_u].t()
        if not use_argmax:
            uniform = torch.rand_like(score_block).clamp(min=1e-10, max=1 - 1e-10)
            score_block = score_block - torch.log(-torch.log(uniform))
        block_max, block_argmax = score_block.max(dim=1)
        block_argmax = block_argmax + v_start
        update_mask = block_max > best_val
        best_val = torch.where(update_mask, block_max, best_val)
        best_idx = torch.where(update_mask, block_argmax, best_idx)
    return best_idx.reshape(B, chunk)

def pushforward_chunk_batched(
    X_theta_chunk: torch.Tensor,  # [B, V, chunk]
    s: float,
    block_params: Optional[List[Tuple[float, torch.Tensor, torch.Tensor, int]]],  # (mu_b, weights_b, lam_b, N_b)
    incoming_ptr: torch.Tensor,       # [V+1] CSR row pointers for INCOMING edges
    incoming_src: torch.Tensor,       # [E] source tokens for incoming edges
    incoming_slot: torch.Tensor,      # [E] slot positions for incoming edges
    buffer_manager: BufferManager,
    csr_cache: Optional[SparseCSRMatmulCache] = None,
) -> torch.Tensor:  # Y_s_chunk [B, V, chunk]
    B, V, chunk = X_theta_chunk.shape
    device = X_theta_chunk.device
    dtype = X_theta_chunk.dtype
    if block_params is None:
        return X_theta_chunk.clone()
    X_flat = X_theta_chunk.transpose(1, 2).reshape(-1, V).t()  # [V, B*chunk]
    Y_flat = X_flat.clone()  # [V, B*chunk]
    Z_buffer = buffer_manager.get_buffer('Z_uniformization', (V, B * chunk), device, dtype)
    Z_next   = buffer_manager.get_buffer('Z_uniformization_next', (V, B * chunk), device, dtype)
    sem_buf   = buffer_manager.get_buffer('F_sem_out', (V, B * chunk), device, dtype)
    uni_buf   = buffer_manager.get_buffer('U_out', (V, B * chunk), device, dtype)
    total_buf = buffer_manager.get_buffer('U_total', (1, B * chunk), device, dtype)
    for b_idx, (mu_b, weights_b, lam_b, N_b) in enumerate(block_params):
        uniformization_apply_block_inplace(
            Y_flat, mu_b, weights_b, lam_b,
            incoming_ptr, incoming_src, incoming_slot,
            Z_buffer, Z_next,
            sem_buf, uni_buf, total_buf,
            N=N_b,
            transpose=DEFAULT_TRANSPOSE,
            csr_cache=csr_cache,
        )
    Y_s_chunk = Y_flat.t().reshape(B, chunk, V).transpose(1, 2)
    return Y_s_chunk

def reverse_step_streaming(
    x_t: torch.Tensor,
    t: float,
    s: float,
    K: int,
    model: Any,
    forward_process: Any,
    schedule: Any,
    knn_idx: torch.Tensor,
    base_score: torch.Tensor,
    incoming_ptr: torch.Tensor,
    incoming_src: torch.Tensor,
    incoming_slot: torch.Tensor,
    buffer_manager: BufferManager,
    chunk_size: int = 128,
    vocab_block_size: int = 50259,
    step_idx: int = 0,
    poisson_eps_mode: str = "mu2_overK",
    poisson_eps_coeff: float = 1.0,
    poisson_eps_rel: float = 1e-6,
    poisson_N_max: int = 256,
    sampling_dtype: torch.dtype = torch.float32,
    last_step_argmax: bool = False,
    freeze_mu_star: float = 1.0,
) -> torch.Tensor:
    B, L = x_t.shape
    V = knn_idx.shape[0]
    device = x_t.device
    Delta = t - s
    full_pf_params = precompute_pushforward_params(
        s=t,
        K=K,
        schedule=schedule,
        forward_process=forward_process,
        base_score=base_score,
        knn_idx=knn_idx,
        mu_star=freeze_mu_star,
        device=device,
        poisson_eps_mode=poisson_eps_mode,
        poisson_eps_coeff=poisson_eps_coeff,
        poisson_eps_rel=poisson_eps_rel,
        poisson_N_max=poisson_N_max,
        sampling_dtype=sampling_dtype,
    ) or []
    s_mu = mu_interval(0.0, s, schedule, device)
    step_pf_params = []
    mu_prefix = 0.0
    for block in full_pf_params:
        mu_prefix += float(block[0])
        if mu_prefix <= s_mu + 1e-12:
            step_pf_params.append(block)
        else:
            break
    t_tensor = torch.tensor(t, device=device, dtype=sampling_dtype)
    alpha_t = schedule.alpha_t(t_tensor)
    sigma_t = model._sigma_from_alphat(alpha_t)
    if sigma_t.ndim == 0:
        sigma_t = sigma_t.view(1, 1)
    elif sigma_t.ndim == 1:
        sigma_t = sigma_t.view(-1, 1)
    if B > 1 and sigma_t.shape[0] == 1:
        sigma_t = sigma_t.expand(B, 1)
    model_out = model.forward(xt=x_t, sigma=sigma_t)
    tbar = 0.5 * (t + s)
    tbar_tensor = torch.tensor(tbar, device=device, dtype=sampling_dtype)
    alpha_bar = schedule.alpha_t(tbar_tensor)
    beta_bar = float(forward_process._exponent(alpha_bar))
    lambda_bar_f = float(forward_process._lambda(tbar_tensor))
    mu_step = mu_interval(s, t, schedule, device)
    if poisson_eps_mode == "delta2":
        eps_step = poisson_eps_coeff * (Delta ** 2)
    elif poisson_eps_mode == "mu2_overK":
        eps_step = poisson_eps_coeff * (mu_step ** 2) / max(1, K)
    else:
        eps_step = poisson_eps_coeff * Delta
    N_step = compute_truncation(mu_step, eps_abs=eps_step, eps_rel=poisson_eps_rel, N_max=poisson_N_max)
    weights_bar: Optional[torch.Tensor] = None
    if lambda_bar_f < 1.0 - 1e-12:
        weights_bar = F.softmax((beta_bar * base_score).to(sampling_dtype), dim=-1)
    x_s = torch.empty_like(x_t)
    csr_cache = SparseCSRMatmulCache(incoming_ptr, incoming_src, incoming_slot, V, device, sampling_dtype)
    for start in range(0, L, chunk_size):
        end = min(start + chunk_size, L)
        actual_chunk = end - start
        logits_chunk = model_out[:, start:end, :]
        X_theta_chunk = F.softmax(logits_chunk.to(sampling_dtype), dim=-1).transpose(1, 2).contiguous()
        x_t_chunk = x_t[:, start:end]
        X_theta_chunk = _debias_x0_posterior_with_xt_likelihood(
            X_theta_chunk=X_theta_chunk,
            x_t_chunk=x_t_chunk,
            t=t,
            K=K,
            schedule=schedule,
            forward_process=forward_process,
            base_score=base_score,
            knn_idx=knn_idx,
            incoming_ptr=incoming_ptr,
            incoming_src=incoming_src,
            incoming_slot=incoming_slot,
            buffer_manager=buffer_manager,
            sampling_dtype=sampling_dtype,
            poisson_eps_mode=poisson_eps_mode,
            poisson_eps_coeff=poisson_eps_coeff,
            poisson_eps_rel=poisson_eps_rel,
            poisson_N_max=poisson_N_max,
            debias_strength=1.0,
            debias_floor=1.0e-12,
            debias_float64_min_floor=1.0e-30,
            precomputed_pf_params_shared_full=full_pf_params,
            freeze_mu_star=freeze_mu_star,
            csr_cache=csr_cache,
        )
        Y_s_chunk = pushforward_chunk_batched(
            X_theta_chunk,
            s,
            step_pf_params,
            incoming_ptr,
            incoming_src,
            incoming_slot,
            buffer_manager,
            csr_cache=csr_cache,
        )
        x_t_flat = x_t_chunk.reshape(-1).clamp(0, V - 1)
        unique_u, inv_u = torch.unique(x_t_flat, return_inverse=True)
        if weights_bar is None:
            dummy_bar = buffer_manager.get_buffer("dummy_weights", (V, base_score.shape[1]), device, sampling_dtype)
            dummy_bar.zero_()
            weights_eff = dummy_bar
        else:
            weights_eff = weights_bar
        logR = compute_step_bayes_log_factors_uniformized(
            unique_u=unique_u,
            inv_u=inv_u,
            mu=mu_step,
            N_step=N_step,
            weights_bar=weights_eff,
            lambda_bar=lambda_bar_f,
            incoming_ptr=incoming_ptr,
            incoming_src=incoming_src,
            incoming_slot=incoming_slot,
            buffer_manager=buffer_manager,
            B=B,
            chunk=actual_chunk,
            device=device,
            dtype=sampling_dtype,
            csr_cache=csr_cache,
        )
        x_s[:, start:end] = sample_chunk_streaming_vocab_with_bayes(
            Y_s_chunk=Y_s_chunk,
            logR=logR,
            inv_u=inv_u,
            vocab_block_size=vocab_block_size,
            sampling_dtype=sampling_dtype,
            use_argmax=last_step_argmax and step_idx == K - 1,
        )
    return x_s

def mu_of_t(t: float, schedule: Any, device: Optional[torch.device] = None) -> float:
    if device is None:
        device = torch.device('cpu')
    t_tensor = torch.tensor(t, dtype=torch.float32, device=device)
    alpha = schedule.alpha_t(t_tensor).clamp(min=1e-10)
    mu = -torch.log(alpha).item()
    return mu

def mu_interval(s: float, t: float, schedule: Any, device: Optional[torch.device] = None) -> float:
    return mu_of_t(t, schedule, device) - mu_of_t(s, schedule, device)

def t_of_mu(mu_target: float, s: float, schedule: Any, device: torch.device, num_iters: int = 30) -> float:
    lo, hi = 0.0, s
    for _ in range(num_iters):
        mid = (lo + hi) / 2.0
        mu_mid = mu_of_t(mid, schedule, device=device)
        if mu_mid < mu_target:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2.0

def build_reverse_timesteps(
    eps: float,
    num_steps: int,
    schedule: Any,
    device: torch.device,
    mode: str = "time",
) -> torch.Tensor:
    if num_steps < 1:
        raise ValueError(f"num_steps must be >= 1, got {num_steps}.")
    if mode == "time":
        return torch.linspace(1.0, float(eps), num_steps + 1, device=device)
    if mode != "mu":
        raise ValueError("reverse_time_grid must be 'time' or 'mu'.")
    eps_f = float(eps)
    mu_hi = mu_of_t(1.0, schedule, device=device)
    mu_lo = mu_of_t(eps_f, schedule, device=device)
    mu_grid = torch.linspace(mu_hi, mu_lo, num_steps + 1, device=device, dtype=torch.float64)
    ts: List[float] = []
    prev_t = 1.0
    for j, mu_j_t in enumerate(mu_grid):
        mu_j = float(mu_j_t.item())
        if j == 0:
            t_j = 1.0
        elif j == num_steps:
            t_j = eps_f
        else:
            t_j = float(t_of_mu(mu_j, 1.0, schedule, device))
            if t_j > prev_t:
                t_j = prev_t
            if t_j < eps_f:
                t_j = eps_f
        ts.append(t_j)
        prev_t = t_j
    ts[-1] = eps_f
    return torch.tensor(ts, device=device, dtype=torch.float32)

def _require_sik_forward_process_api(forward_process: Any) -> None:
    required = ("schedule", "kernel", "_lambda", "_exponent")
    missing = [k for k in required if not hasattr(forward_process, k)]
    if missing:
        raise ValueError(
            "GDDSSIKKNNSampler requires a SIK-compatible forward process with "
            f"{required}. Missing: {missing}. Got: {type(forward_process).__name__}"
        )

def build_freeze_boundaries(
    u0: float,
    u1: float,
    schedule: Any,
    mu_star: float,
    device: torch.device,
) -> List[float]:
    mu0 = mu_of_t(u0, schedule, device=device)
    mu1 = mu_of_t(u1, schedule, device=device)
    num_blocks = max(1, int(math.ceil((mu1 - mu0) / mu_star)))
    boundaries = [u0 + (u1 - u0) * j / num_blocks for j in range(num_blocks + 1)]
    boundaries[-1] = u1
    return boundaries

def build_incoming_adjacency(
    knn_idx: torch.Tensor,
    V: int,
    k: int,
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    src = torch.arange(V, device=device).repeat_interleave(k)  # [V*k]
    dst = knn_idx.view(-1)  # [V*k]
    slot = torch.arange(k, device=device).repeat(V)  # [V*k]
    sorted_idx = torch.argsort(dst, stable=True)
    src_sorted = src[sorted_idx]
    dst_sorted = dst[sorted_idx]
    slot_sorted = slot[sorted_idx]
    counts = torch.bincount(dst_sorted, minlength=V)  # [V]
    incoming_ptr = torch.cat([
        torch.zeros(1, device=device, dtype=torch.long),
        counts.cumsum(0, dtype=torch.long)
    ])  # [V+1]
    return incoming_ptr, src_sorted, slot_sorted

def compute_base_score_from_kernel(kernel: Any, device: torch.device) -> torch.Tensor:
    V = kernel.m
    k = kernel._knn_indices.shape[1]
    knn_idx = kernel._knn_indices.to(device)  # [V, k]
    knn_dist = kernel._knn_distances.to(device)  # [V, k]
    logK = -(knn_dist / kernel.epsilon)  # [V, k]
    if kernel.gamma > 0:
        n = kernel.get_n(device)  # [V]
        log_n_i = torch.log(n).unsqueeze(1)  # [V, 1]
        log_n_j = torch.log(n[knn_idx])  # [V, k]
        base_score = logK - log_n_i - log_n_j
    else:
        base_score = logK
    self_mask = (knn_idx == torch.arange(V, device=device).unsqueeze(1))
    base_score = base_score.masked_fill(self_mask, float('-inf'))
    return base_score

def _build_block_params_from_boundaries(
    boundaries: List[float],
    K: int,
    schedule: Any,
    forward_process: Any,
    base_score: torch.Tensor,
    knn_idx: torch.Tensor,
    device: torch.device,
    *,
    poisson_eps_mode: str = "mu2_overK",
    poisson_eps_coeff: float = 1.0,
    poisson_eps_rel: float = 1e-6,
    poisson_N_max: int = 256,
    sampling_dtype: torch.dtype = torch.float32,
) -> Optional[List[Tuple[float, torch.Tensor, torch.Tensor, int]]]:
    if len(boundaries) < 2:
        return None
    total_span = max(0.0, float(boundaries[-1]) - float(boundaries[0]))
    if total_span <= 1e-12:
        return None
    mu_total = mu_interval(float(boundaries[0]), float(boundaries[-1]), schedule, device)
    if mu_total <= 1e-10:
        return None
    Delta = 1.0 / K
    B_blocks = len(boundaries) - 1
    if poisson_eps_mode == "delta2":
        eps_total = poisson_eps_coeff * (Delta ** 2)
    elif poisson_eps_mode == "mu2_overK":
        eps_total = poisson_eps_coeff * (mu_total ** 2) / max(1, K)
    else:
        eps_total = poisson_eps_coeff * Delta
    blocks_meta = []
    for b in range(B_blocks):
        u_b = float(boundaries[b])
        u_b1 = float(boundaries[b + 1])
        mu_b = mu_interval(u_b, u_b1, schedule, device)
        m_b = 0.5 * (u_b + u_b1)
        t_mid = torch.tensor(m_b, device=device, dtype=sampling_dtype)
        alpha_eff = schedule.alpha_t(t_mid)
        beta_b = forward_process._exponent(alpha_eff)
        lam_b = forward_process._lambda(t_mid)
        lam_b_f = float(lam_b)
        V = base_score.shape[0]
        k = knn_idx.shape[1]
        if lam_b_f >= 1.0 - 1e-12:
            weights_b = torch.zeros((V, k), device=device, dtype=sampling_dtype)
        else:
            weights_b = F.softmax((beta_b * base_score).to(sampling_dtype), dim=-1)
        blocks_meta.append((mu_b, weights_b, lam_b))
    eps_blocks = [eps_total / B_blocks for _ in blocks_meta]
    block_params = []
    for (mu_b, weights_b, lam_b), eps_b in zip(blocks_meta, eps_blocks):
        N_b = compute_truncation(mu_b, eps_abs=eps_b, eps_rel=poisson_eps_rel, N_max=poisson_N_max)
        block_params.append((mu_b, weights_b, lam_b, N_b))
    return block_params

def precompute_pushforward_params(
    s: float,
    K: int,
    schedule: Any,
    forward_process: Any,
    base_score: torch.Tensor,
    knn_idx: torch.Tensor,
    mu_star: float = 1.0,
    device: Optional[torch.device] = None,
    *,
    poisson_eps_mode: str = "mu2_overK",
    poisson_eps_coeff: float = 1.0,
    poisson_eps_rel: float = 1e-6,
    poisson_N_max: int = 256,
    sampling_dtype: torch.dtype = torch.float32,
) -> Optional[List[Tuple[float, torch.Tensor, torch.Tensor, int]]]:
    if device is None:
        device = base_score.device
    boundaries = build_freeze_boundaries(
        0.0, s, schedule,
        mu_star=mu_star,
        device=device,
    )
    block_params = _build_block_params_from_boundaries(
        boundaries,
        K,
        schedule,
        forward_process,
        base_score,
        knn_idx,
        device,
        poisson_eps_mode=poisson_eps_mode,
        poisson_eps_coeff=poisson_eps_coeff,
        poisson_eps_rel=poisson_eps_rel,
        poisson_N_max=poisson_N_max,
        sampling_dtype=sampling_dtype,
    )
    return block_params

class GDDSSIKKNNSampler(Sampler):

    def __init__(
        self,
        config: Any,
        forward_process: Optional[Any] = None,
        mu_star: float = 1.0,
        chunk_size: int = 128,
        vocab_block_size: int = 50259,
        poisson_eps_mode: str = "mu2_overK",
        poisson_eps_coeff: float = 1.0,
        poisson_eps_rel: float = 1e-6,
        poisson_N_max: int = 256,
        freeze_mu_star: float = 1.0,
        reverse_time_grid: str = "time",
        use_float64: bool = False,
        last_step_argmax: bool = False,
        **_ignored: Any,
    ):
        sampler_cfg = None
        try:
            sampler_cfg = getattr(getattr(config, "sampling", None), "sampler", None)
        except Exception:
            sampler_cfg = None
        sampler_target = str(getattr(sampler_cfg, "_target_", "")) if sampler_cfg is not None else ""
        if sampler_cfg is not None and sampler_target.endswith("GDDSSIKKNNSampler"):
            mu_star = sampler_cfg.get("mu_star", mu_star)
            chunk_size = sampler_cfg.get("chunk_size", chunk_size)
            vocab_block_size = sampler_cfg.get("vocab_block_size", vocab_block_size)
            poisson_eps_mode = sampler_cfg.get("poisson_eps_mode", poisson_eps_mode)
            poisson_eps_coeff = sampler_cfg.get("poisson_eps_coeff", poisson_eps_coeff)
            poisson_eps_rel = sampler_cfg.get("poisson_eps_rel", poisson_eps_rel)
            poisson_N_max = sampler_cfg.get("poisson_N_max", poisson_N_max)
            freeze_mu_star = sampler_cfg.get("freeze_mu_star", freeze_mu_star)
            reverse_time_grid = sampler_cfg.get("reverse_time_grid", reverse_time_grid)
            use_float64 = sampler_cfg.get("use_float64", use_float64)
            last_step_argmax = sampler_cfg.get("last_step_argmax", last_step_argmax)
        self.config = config
        self.forward_process = forward_process
        self.mu_star = mu_star
        self.chunk_size = chunk_size
        self.vocab_block_size = vocab_block_size
        self.poisson_eps_mode = poisson_eps_mode
        self.poisson_eps_coeff = poisson_eps_coeff
        self.poisson_eps_rel = poisson_eps_rel
        self.poisson_N_max = poisson_N_max
        self.freeze_mu_star = freeze_mu_star
        self.reverse_time_grid = str(reverse_time_grid)
        if self.reverse_time_grid not in ("time", "mu"):
            raise ValueError("reverse_time_grid must be 'time' or 'mu'.")
        self.use_float64 = bool(use_float64)
        self._sampling_dtype = torch.float64 if self.use_float64 else torch.float32
        self.last_step_argmax = last_step_argmax
        self.pushforward_xt_likelihood_debias = True
        self.pushforward_xt_likelihood_debias_strength = 1.0
        self.pushforward_xt_likelihood_debias_floor = 1.0e-12
        self.pushforward_xt_likelihood_debias_float64_min_floor = 1.0e-30
        self._buffer_manager = BufferManager()
        self._knn_idx: Optional[torch.Tensor] = None
        self._base_score: Optional[torch.Tensor] = None
        self._incoming_ptr: Optional[torch.Tensor] = None
        self._incoming_src: Optional[torch.Tensor] = None
        self._incoming_slot: Optional[torch.Tensor] = None
        self._V: Optional[int] = None
        self._k: Optional[int] = None
        self._cached_device: Optional[torch.device] = None

    def _get_forward_process(self, model: Any) -> Any:
        if self.forward_process is not None:
            return self.forward_process
        fp = getattr(model, '_forward_process', None) or getattr(model, 'forward_process', None)
        if fp is None and hasattr(model, '_setup_forward_process'):
            model._setup_forward_process()
            fp = getattr(model, '_forward_process', None) or getattr(model, 'forward_process', None)
        if fp is None:
            raise ValueError(
                "forward_process not provided and not found in model. "
                "For GDDS, ensure the model was loaded with config from the checkpoint "
                "(e.g. config=ckpt['hyper_parameters']['config']) so _setup_forward_process can build the FP."
            )
        _require_sik_forward_process_api(fp)
        return fp

    def _cache_kernel_data(self, forward_process: Any, device: torch.device) -> None:
        if self._cached_device == device and self._knn_idx is not None:
            return
        kernel = forward_process.kernel
        if hasattr(kernel, '_knn_indices'):
            knn_idx = kernel._knn_indices.to(device)
            self._knn_idx = knn_idx
            self._k = knn_idx.shape[1]
        else:
            raise ValueError("Kernel must have _knn_indices attribute")
        self._base_score = compute_base_score_from_kernel(kernel, device)
        self._V = kernel.m
        self._cached_device = device

    def _build_incoming_adjacency(self, device: torch.device) -> None:
        if self._cached_device == device and self._incoming_ptr is not None:
            return
        assert self._knn_idx is not None and self._V is not None and self._k is not None
        self._incoming_ptr, self._incoming_src, self._incoming_slot = build_incoming_adjacency(
            self._knn_idx, self._V, self._k, device
        )
        self._cached_device = device

    def clear_buffers(self):
        self._buffer_manager.clear()
    @torch.no_grad()

    def generate(
        self,
        model: Any,
        *,
        num_samples: int,
        num_steps: int,
        eps: float,
        inject_bos: bool,
        step_callback: Optional[Callable[[int, torch.Tensor], None]] = None,
    ) -> torch.Tensor:
        forward_process = self._get_forward_process(model)
        schedule = forward_process.schedule
        device = next(model.parameters()).device
        self._cache_kernel_data(forward_process, device)
        self._build_incoming_adjacency(device)
        x = model.prior_sample(num_samples, model.num_tokens)  # [B, L]
        if inject_bos:
            x[:, 0] = model.tokenizer.bos_token_id
        effective_steps = num_steps
        timesteps = build_reverse_timesteps(
            eps=eps,
            num_steps=num_steps,
            schedule=schedule,
            device=device,
            mode=self.reverse_time_grid,
        )
        assert self._knn_idx is not None, "knn_idx not cached"
        assert self._base_score is not None, "base_score not cached"
        assert self._incoming_ptr is not None, "incoming_ptr not built"
        assert self._incoming_src is not None, "incoming_src not built"
        assert self._incoming_slot is not None, "incoming_slot not built"
        for step_idx in range(effective_steps):
            t_val = timesteps[step_idx].item()
            s_val = timesteps[step_idx + 1].item()
            x = reverse_step_streaming(
                x, t_val, s_val, effective_steps, model, forward_process, schedule,
                self._knn_idx, self._base_score,
                self._incoming_ptr, self._incoming_src, self._incoming_slot,
                self._buffer_manager,
                chunk_size=self.chunk_size,
                vocab_block_size=self.vocab_block_size,
                step_idx=step_idx,
                poisson_eps_mode=self.poisson_eps_mode,
                poisson_eps_coeff=self.poisson_eps_coeff,
                poisson_eps_rel=self.poisson_eps_rel,
                poisson_N_max=self.poisson_N_max,
                sampling_dtype=self._sampling_dtype,
                last_step_argmax=self.last_step_argmax,
                freeze_mu_star=self.freeze_mu_star,
            )  # [B, L]
            if step_callback is not None:
                step_callback(step_idx, x)
        return x
