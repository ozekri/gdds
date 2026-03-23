import math
import typing

import einops
import omegaconf
import torch
import torch.nn as nn
import torch.nn.functional as F

try:  # flash-attn is optional but recommended
  import flash_attn
  import flash_attn.layers.rotary
except (ImportError, RuntimeError):
  flash_attn = None  # type: ignore

try:  # flex-attention is optional (PyTorch 2.4+)
  from torch.nn.attention.flex_attention import flex_attention, create_block_mask
  FLEX_ATTN_AVAILABLE = True
except (ImportError, RuntimeError):
  flex_attention = None  # type: ignore
  create_block_mask = None  # type: ignore
  FLEX_ATTN_AVAILABLE = False
from .common import (
  bias_dropout_add_scale,
  get_bias_dropout_add_scale,
  bias_dropout_add_scale_fused_train,
  bias_dropout_add_scale_fused_inference,
  modulate,
  modulate_fused,
  Rotary,
  rotate_half,
  split_and_apply_rotary_pos_emb,
  apply_rotary_pos_emb,
  apply_rotary_pos_emb_single,
  LayerNorm,
  residual_linear,
  TimestepEmbedder,
  EmbeddingLayer,
  LabelEmbedder,
  DDiTBlock,
  DDiTBlockCausal,
  DDiTFinalLayer,
  sdpa_attention_unmasked,
  sdpa_attention_masked,
  flash_varlen_attention_qkvpacked,
  flash_cross_attention,
  supports_flash_attention,
  supports_flex_attention,
)

# Flags required to enable jit fusion kernels
torch._C._jit_set_profiling_mode(False)
torch._C._jit_set_profiling_executor(False)
torch._C._jit_override_can_fuse_on_cpu(True)
torch._C._jit_override_can_fuse_on_gpu(True)


def gated_update(
    x_old: torch.Tensor,
    x_new: torch.Tensor, 
    query_enabled: torch.Tensor | None,
) -> torch.Tensor:
    """Gated update: only update positions where query_enabled is True.
    
    This is CRITICAL for sampling correctness: non-queried positions must remain
    EXACTLY unchanged (bitwise identical) after a block, including after attention
    AND MLP residuals. Without this, FFN/adaLN/residual can drift non-queried tokens
    and corrupt the key/value bank for process-attention.
    
    Args:
        x_old: [B, S, D] tensor before the sublayer
        x_new: [B, S, D] tensor after the sublayer
        query_enabled: None (no gating, return x_new) or [B, S] bool tensor
                       where True = position is allowed to update
    
    Returns:
        Gated output where disabled positions keep x_old values exactly.
    """
    if query_enabled is None:
        return x_new
    
    # Convert bool [B, S] to float [B, S, 1] for broadcasting
    # Use the same dtype as x_new for numerical stability
    gate = query_enabled.to(dtype=x_new.dtype, device=x_new.device).unsqueeze(-1)
    return x_new * gate + x_old * (1.0 - gate)


def make_group_self_attn_mask(group_idxs):
  # Return shape: N x L x L
  return group_idxs[:, None, :] == group_idxs[:, :, None]


# Training
def make_group_cross_attn_mask(group_idxs):
  # Return N x L x L
  return group_idxs[:, None, :] != group_idxs[:, :, None]


# Inference
def make_inference_self_attn_mask(seq_len, concrete_lengths):
  arrange = torch.arange(seq_len, device=concrete_lengths.device)
  mask = arrange[None, :] < concrete_lengths[:, None]
  mask = mask[:, None, :].repeat(1, seq_len, 1)
  return mask


def make_inference_cross_attn_mask(
    keys_tensor_length, 
    queries_tensor_length,
    concrete_lengths_keys,):
  """
  Queries positions == noisy positions
  Key positions == denoised positions
  Concrete length: number of denoised tokens in each 
                        element of the batch.
  """
  arrange = torch.arange(keys_tensor_length, device=concrete_lengths_keys.device)
  mask = arrange[None] < concrete_lengths_keys[:, None]  # BS x KV_LEN
  mask = mask[:, None, :]  # BS x 1 x KV_LEN
  mask = mask.repeat(1, queries_tensor_length, 1)  # BS x Q_LEN x KV_LEN
  return mask


def get_sinusoidal_embedding(idxs, dim, base=10_000):
  device = idxs.device
  denominator = base ** (2 * torch.arange(dim // 2, 
                                          device=device) / dim)
  arg = idxs[:, None] / denominator[None, :]
  cos = torch.cos(arg)
  sin = torch.sin(arg)
  out = torch.cat([cos, sin], dim=-1)
  return out


@torch.jit.script
def _partition_mean_train(x, group_idxs, group: int):
  mask = (group_idxs == group)  # BS x L
  group_div = mask.sum(-1)  # BS
  group_div = torch.where(group_div == 0, 1, group_div)
  out = (x * mask[..., None]).sum(1) / group_div[:, None]  # BS x H
  return mask, out  # (BS x L, BS x H)


@torch.jit.script
def _partition_logsumexp_train(x, group_idxs, group: int):
  mask = (group_idxs == group)  # BS x L
  out = torch.where(mask[..., None], x, -float('inf'))  # BS x L x H
  out = torch.logsumexp(out, dim=1)  # BS x H
  out = torch.where(out.isinf(), 0.0, out)
  return mask, out  # (BS x L, BS x H)


@torch.jit.script
def _partition_mean_inference(x, concrete_lengths):
  arrange = torch.arange(x.shape[1], device=x.device)[None, :]  # 1 x L
  mask = (arrange < concrete_lengths[:, None])  # BS x L
  group_div = mask.sum(1)  # BS
  group_div = torch.where(group_div == 0, 1, group_div)
  out = (x * mask[..., None]).sum(1) / group_div[:, None]  # BS x H
  return out  # BS x H


@torch.jit.script
def _partition_logsumexp_inference(x, concrete_lengths):
  arrange = torch.arange(x.shape[1], device=x.device)[None, :]  # 1 x L
  mask = arrange < concrete_lengths[:, None]  # BS x L
  out = torch.where(mask[..., None], x, -float('inf'))  # BS x L x H
  out = torch.logsumexp(out, dim=1)  # BS x H
  out = torch.where(out.isinf(), 0.0, out)  # BS x H
  return out  # BS x H


@torch.jit.script
def _index_rotary(source_tensor, index):
  # source shape: 1 x L x 3 x 1 x H/2
  # index shape: BS x L
  # out shape: BS x L x 3 x 1 x H/2
  index = index[..., None, None, None]  # BS x L x 1 x 1 x 1
  # BS x L x 3 x 1 x H/2
  index = index.repeat(1,1, source_tensor.shape[2], 
                       source_tensor.shape[3], 
                       source_tensor.shape[4])
  # BS x L x 3 x 1 x H/2
  source_tensor = source_tensor.repeat(index.shape[0],1,1,1,1)
  out = torch.gather(source_tensor, dim=1, index=index)
  return out


@torch.jit.script
def _index_freqs_swap(pos_freqs, positions):
  freqs = pos_freqs[None, positions]  # 1 x L x H
  freqs = pos_freqs[None].repeat(positions.shape[0], 1, 1)
  positions = positions[..., None].repeat(1, 1, freqs.shape[-1])
  freqs = torch.gather(freqs, dim=1, index=positions)
  return freqs


## Moved to common: sdpa_attention_unmasked, sdpa_attention_masked


#################################################################################
#                                  Layers                                       #
#################################################################################


#################################################################################
#               Embedding Layers for Timesteps and Class Labels                 #
#################################################################################


#################################################################################
#                                 Core Model                                    #
#################################################################################

class DDiTBlock(nn.Module):
  def __init__(self, dim, n_heads, adaLN,
               cond_dim=None, mlp_ratio=4,
               dropout=0.1, attn_backend='auto'):
    super().__init__()
    self.n_heads = n_heads
    self.adaLN = adaLN
    self.attn_backend = attn_backend

    self.norm1 = LayerNorm(dim)
    self.attn_qkv = nn.Linear(dim, 3 * dim, bias=False)
    self.attn_out = nn.Linear(dim, dim, bias=False)
    self.dropout1 = nn.Dropout(dropout)

    self.norm2 = LayerNorm(dim)
    self.mlp = nn.Sequential(
      nn.Linear(dim, mlp_ratio * dim, bias=True),
      nn.GELU(approximate='tanh'),
      nn.Linear(mlp_ratio * dim, dim, bias=True))
    self.dropout2 = nn.Dropout(dropout)
    self.dropout = dropout

    if self.adaLN:
      self.adaLN_modulation = nn.Linear(cond_dim, 6 * dim)
      self.adaLN_modulation.weight.data.zero_()
      b = self.adaLN_modulation.bias.data
      b.zero_()
      # Initialize gate biases to small positive values to speed escaping identity basin
      # gate_msa is at index 2*dim:3*dim, gate_mlp is at 5*dim:6*dim
      b[2*dim:3*dim] = 0.5  # gate_msa
      b[5*dim:6*dim] = 0.5   # gate_mlp


  def _get_bias_dropout_scale(self):
    if self.training:
      return bias_dropout_add_scale_fused_train
    else:
      return bias_dropout_add_scale_fused_inference


  def forward(self, x, t_cond, rotary_cos_sin, self_attn_mask, query_enabled=None):
    """Forward pass with optional gated updates for sampling correctness.
    
    Args:
        x: [B, S, D] input tensor
        t_cond: [B, cond_dim] or [B, S, cond_dim] time conditioning
        rotary_cos_sin: RoPE embeddings
        self_attn_mask: attention mask
        query_enabled: [B, S] bool or None. If provided, only positions where
                       query_enabled=True are allowed to update. Other positions
                       remain exactly unchanged after this block.
    """
    bias_dropout_scale_fn = self._get_bias_dropout_scale()

    # Save original state BEFORE any updates (for gating)
    x_before_attn = x
    
    x_normed = self.norm1(x)

    if self.adaLN:
      # Support both [B, cond_dim] (global) and [B, n, cond_dim] (per-token) conditioning
      if t_cond is not None and t_cond.ndim == 3:  # [B, n, cond_dim]
        t_cond_mod = self.adaLN_modulation(t_cond)  # [B, n, 6*dim]
        (shift_msa, scale_msa, gate_msa, shift_mlp,
         scale_mlp, gate_mlp) = t_cond_mod.chunk(6, dim=2)  # Each [B, n, dim]
      else:  # [B, cond_dim] or None
        if t_cond is not None:
          t_cond_mod = self.adaLN_modulation(t_cond)  # [B, 6*dim]
          (shift_msa, scale_msa, gate_msa, shift_mlp,
           scale_mlp, gate_mlp) = t_cond_mod[:, None].chunk(6, dim=2)  # Each [B, 1, dim]
        else:
          # No conditioning - use zeros (shouldn't happen if adaLN=True, but handle gracefully)
          shift_msa = scale_msa = gate_msa = shift_mlp = scale_mlp = gate_mlp = None
      if t_cond is not None:
        x_normed = modulate_fused(x_normed, shift_msa, scale_msa)

    qkv = einops.rearrange(
      self.attn_qkv(x_normed),
      'b s (three h d) -> b s three h d',
      three=3,
      h=self.n_heads)
    q, k, v = split_and_apply_rotary_pos_emb(qkv, rotary_cos_sin)
    
    # Try FlexAttention if explicitly requested and mask is present
    # Note: For arbitrary boolean masks, SDPA is often more efficient than FlexAttention
    # FlexAttention works best for structured masks (causal, block-diagonal, etc.)
    use_flex = (
      self.attn_backend == 'flex' and  # Only use if explicitly requested
      FLEX_ATTN_AVAILABLE and
      self_attn_mask is not None
    )
    
    if use_flex:
      # Convert boolean mask to BlockMask format for FlexAttention
      # FlexAttention expects (B, H, S, D) format
      q_flex = einops.rearrange(q, 'b s h d -> b h s d')
      k_flex = einops.rearrange(k, 'b s h d -> b h s d')
      v_flex = einops.rearrange(v, 'b s h d -> b h s d')
      
      # Create mask function for FlexAttention
      # Convert to boolean if needed
      mask_bool = self_attn_mask if self_attn_mask.dtype == torch.bool else self_attn_mask > -1e3
      # Ensure mask is on the same device
      mask_bool = mask_bool.to(q.device)
      
      # Create a closure that captures the mask tensor
      def mask_mod(b, h, q_idx, kv_idx):
        # FlexAttention may pass tensors or scalars, handle both
        if isinstance(b, torch.Tensor):
          batch_idx = b
        else:
          batch_idx = torch.tensor(b, device=mask_bool.device)
        if isinstance(q_idx, torch.Tensor):
              q_idx_t = q_idx
        else:
          q_idx_t = torch.tensor(q_idx, device=mask_bool.device)
        if isinstance(kv_idx, torch.Tensor):
          kv_idx_t = kv_idx
        else:
          kv_idx_t = torch.tensor(kv_idx, device=mask_bool.device)
        return mask_bool[batch_idx, q_idx_t, kv_idx_t]
      
      try:
        block_mask = create_block_mask(
          mask_mod,
          B=q.shape[0],
          H=self.n_heads,
          Q_LEN=q.shape[1],
          KV_LEN=k.shape[1],
        )
        x_flex = flex_attention(q_flex, k_flex, v_flex, block_mask=block_mask)
        x_attn = einops.rearrange(x_flex, 'b h s d -> b s (h d)')
      except Exception as e:
        # Fallback to SDPA if FlexAttention fails
        # Note: FlexAttention may not be optimal for arbitrary boolean masks
        # SDPA handles them efficiently with its optimized kernels
        import warnings
        warnings.warn(f"FlexAttention failed, falling back to SDPA: {e}", RuntimeWarning)
        x_attn = sdpa_attention_masked(q, k, v, self_attn_mask, causal=False)
    else:
      x_attn = sdpa_attention_masked(q, k, v, self_attn_mask, causal=False)

    # Apply attention residual + gating
    if self.adaLN:
      x = bias_dropout_scale_fn(self.attn_out(x_attn),
                                None,
                                gate_msa,
                                x_before_attn,
                                self.dropout)
    else:
      scale = torch.ones(1, device=x_attn.device, dtype=x_attn.dtype)
      x = bias_dropout_scale_fn(
        self.attn_out(x_attn), None, scale, x_before_attn, self.dropout)
    
    # GATE after attention residual: freeze non-queried positions
    x = gated_update(x_before_attn, x, query_enabled)
    
    # Save state before MLP
    x_before_mlp = x
    
    # MLP sublayer
    if self.adaLN:
      x = bias_dropout_scale_fn(
        self.mlp(modulate_fused(
          self.norm2(x), shift_mlp, scale_mlp)),
        None, gate_mlp, x_before_mlp, self.dropout)
    else:
      scale = torch.ones(1, device=x.device, dtype=x.dtype)
      x = bias_dropout_scale_fn(
        self.mlp(self.norm2(x)), None, scale, x_before_mlp, self.dropout)
    
    # GATE after MLP residual: freeze non-queried positions
    x = gated_update(x_before_mlp, x, query_enabled)
    
    return x


class CrossAttnDDiTBlock(nn.Module):
  def __init__(self, dim, n_heads, adaLN, cond_dim=None, 
               mlp_ratio=4, dropout=0.1, attn_backend='auto'):
    super().__init__()
    self.n_heads = n_heads
    self.adaLN = adaLN
    self.attn_backend = attn_backend

    self.q_norm1 = LayerNorm(dim)
    self.kv_norm1 = LayerNorm(dim)

    self.norm2 = LayerNorm(dim)
    self.attn_q = nn.Linear(dim, dim, bias=False)
    self.attn_kv = nn.Linear(dim, 2 * dim, bias=False)
    self.attn_out = nn.Linear(dim, dim, bias=False)
    self.dropout1 = nn.Dropout(dropout)

    self.mlp = nn.Sequential(
      nn.Linear(dim, mlp_ratio * dim, bias=True),
      nn.GELU(approximate='tanh'),
      nn.Linear(mlp_ratio * dim, dim, bias=True))
    self.dropout2 = nn.Dropout(dropout)
    self.dropout = dropout

    if self.adaLN:
      self.adaLN_modulation = nn.Linear(cond_dim, 6 * dim)
      self.adaLN_modulation.weight.data.zero_()
      b = self.adaLN_modulation.bias.data
      b.zero_()
      # Initialize gate biases to small positive values to speed escaping identity basin
      # gate_msa is at index 2*dim:3*dim, gate_mlp is at 5*dim:6*dim
      b[2*dim:3*dim] = 0.5  # gate_msa
      b[5*dim:6*dim] = 0.5   # gate_mlp

  def _get_bias_dropout_scale(self):
    if self.training:
      return bias_dropout_add_scale_fused_train
    else:
      return bias_dropout_add_scale_fused_inference

  def _apply_cross_attention(self, q, k, v, attn_mask):
    """Apply cross-attention with flex, flash, or SDPA backend, optimized for masked attention."""
    # Try FlexAttention if explicitly requested and mask is present
    # Note: For arbitrary boolean masks, SDPA is often more efficient than FlexAttention
    # FlexAttention works best for structured masks (causal, block-diagonal, etc.)
    use_flex = (
      self.attn_backend == 'flex' and  # Only use if explicitly requested
      FLEX_ATTN_AVAILABLE and
      attn_mask is not None
    )
    
    if use_flex:
      # Convert to (B, H, S, D) format for FlexAttention
      q_flex = einops.rearrange(q, 'b s h d -> b h s d')
      k_flex = einops.rearrange(k, 'b s h d -> b h s d')
      v_flex = einops.rearrange(v, 'b s h d -> b h s d')
      
      # Create mask function for FlexAttention
      mask_bool = attn_mask if attn_mask.dtype == torch.bool else attn_mask > -1e3
      mask_bool = mask_bool.to(q.device)
      
      # Create a closure that captures the mask tensor
      def mask_mod(b, h, q_idx, kv_idx):
        # FlexAttention may pass tensors or scalars, handle both
        if isinstance(b, torch.Tensor):
          batch_idx = b
        else:
          batch_idx = torch.tensor(b, device=mask_bool.device)
        if isinstance(q_idx, torch.Tensor):
          q_idx_t = q_idx
        else:
          q_idx_t = torch.tensor(q_idx, device=mask_bool.device)
        if isinstance(kv_idx, torch.Tensor):
          kv_idx_t = kv_idx
        else:
          kv_idx_t = torch.tensor(kv_idx, device=mask_bool.device)
        return mask_bool[batch_idx, q_idx_t, kv_idx_t]
      
      try:
        block_mask = create_block_mask(
          mask_mod,
          B=q.shape[0],
          H=self.n_heads,
          Q_LEN=q.shape[1],
          KV_LEN=k.shape[1],
        )
        x_flex = flex_attention(q_flex, k_flex, v_flex, block_mask=block_mask)
        return einops.rearrange(x_flex, 'b h s d -> b s (h d)')
      except Exception as e:
        # Fallback to SDPA if FlexAttention fails
        # Note: FlexAttention may not be optimal for arbitrary boolean masks
        # SDPA handles them efficiently with its optimized kernels
        import warnings
        warnings.warn(f"FlexAttention failed in cross-attention, falling back to SDPA: {e}", RuntimeWarning)
        pass
    
    # Fallback to Flash Attention (no mask) or SDPA
    use_flash = (
      self.attn_backend == 'flash_attn' or 
      (self.attn_backend == 'auto' and supports_flash_attention())
    )
    # Flash attention requires no mask (or we'd need varlen which is complex for cross-attn)
    if use_flash and attn_mask is None:
      return flash_cross_attention(q, k, v, causal=False)
    # Use SDPA for masked attention (optimized path)
    # SDPA handles additive masks efficiently with its fast kernels
    if attn_mask is None:
      return sdpa_attention_unmasked(q, k, v)
    # For masked attention, ensure mask is in optimal format for SDPA
    # SDPA expects additive masks (0.0 = allowed, large negative = masked)
    return sdpa_attention_masked(q, k, v, attn_mask, causal=False)

  def forward(self, q_x, kv_x, t_cond, rotary_cos_sin_queries, 
              rotary_cos_sin_keys, attn_mask, query_enabled=None):
    """Forward pass with optional gated updates for sampling correctness.
    
    Args:
        q_x: [B, S, D] query input
        kv_x: [B, K, D] key/value input
        t_cond: [B, cond_dim] or [B, S, cond_dim] time conditioning
        rotary_cos_sin_queries: RoPE for queries
        rotary_cos_sin_keys: RoPE for keys
        attn_mask: attention mask
        query_enabled: [B, S] bool or None. If provided, only positions where
                       query_enabled=True are allowed to update. Other positions
                       remain exactly unchanged after this block.
    """
    bias_dropout_scale_fn = self._get_bias_dropout_scale()

    # Save original state BEFORE any updates (for gating)
    x_before_attn = q_x
    
    q_x_normed = self.q_norm1(q_x)
    kv_x = self.kv_norm1(kv_x)
    if self.adaLN:
      # Support both [B, cond_dim] (global) and [B, n, cond_dim] (per-token) conditioning
      if t_cond is not None and t_cond.ndim == 3:  # [B, n, cond_dim]
        t_cond_mod = self.adaLN_modulation(t_cond)  # [B, n, 6*dim]
        (shift_msa, scale_msa, gate_msa, shift_mlp,
         scale_mlp, gate_mlp) = t_cond_mod.chunk(6, dim=2)  # Each [B, n, dim]
      else:  # [B, cond_dim] or None
        if t_cond is not None:
          t_cond_mod = self.adaLN_modulation(t_cond)  # [B, 6*dim]
          (shift_msa, scale_msa, gate_msa, shift_mlp,
           scale_mlp, gate_mlp) = t_cond_mod[:, None].chunk(6, dim=2)  # Each [B, 1, dim]
        else:
          shift_msa = scale_msa = gate_msa = shift_mlp = scale_mlp = gate_mlp = None
      if t_cond is not None:
        q_x_normed = modulate_fused(q_x_normed, shift_msa, scale_msa)

    q = einops.rearrange(
      self.attn_q(q_x_normed),
      'b s (h d) -> b s h d',
      h=self.n_heads)
    kv = einops.rearrange(
      self.attn_kv(kv_x),
      'b s (two h d) -> b s two h d',
      two=2,
      h=self.n_heads)

    k, v = torch.chunk(kv, chunks=2, dim=2)
    k = k[:, :, 0, :]
    v = v[:, :, 0, :]

    q = apply_rotary_pos_emb_single(q, *rotary_cos_sin_queries)
    k = apply_rotary_pos_emb_single(k, *rotary_cos_sin_keys)
    x_attn = self._apply_cross_attention(q, k, v, attn_mask)

    # Apply attention residual + gating
    if self.adaLN:
      x = bias_dropout_scale_fn(self.attn_out(x_attn),
                                None,
                                gate_msa,
                                x_before_attn,
                                self.dropout)
    else:
      scale = torch.ones(1, device=x_attn.device, dtype=x_attn.dtype)
      x = bias_dropout_scale_fn(
        self.attn_out(x_attn), None, scale, x_before_attn, self.dropout)
    
    # GATE after attention residual: freeze non-queried positions
    x = gated_update(x_before_attn, x, query_enabled)
    
    # Save state before MLP
    x_before_mlp = x
    
    # MLP sublayer
    if self.adaLN:
      x = bias_dropout_scale_fn(
        self.mlp(modulate_fused(
          self.norm2(x), shift_mlp, scale_mlp)),
        None, gate_mlp, x_before_mlp, self.dropout)
    else:
      scale = torch.ones(1, device=x.device, dtype=x.dtype)
      x = bias_dropout_scale_fn(
        self.mlp(self.norm2(x)), None, scale, x_before_mlp, self.dropout)
    
    # GATE after MLP residual: freeze non-queried positions
    x = gated_update(x_before_mlp, x, query_enabled)
    
    return x


class Encoder(nn.Module):
  def __init__(self, n_blocks, dim, n_heads, cond_dim, mlp_ratio, 
               dropout, adaLN):
    super().__init__()
    self.blocks = nn.ModuleList([DDiTBlock(dim, n_heads, adaLN, 
                                           cond_dim, mlp_ratio, 
                                           dropout) 
                                for _ in range(n_blocks)])

  def forward(self, x, t_cond, rotary_cos_sin, self_attn_mask):
    for layer in self.blocks:
      x = layer(x, t_cond, rotary_cos_sin, self_attn_mask)
    return x

  
class GroupSwapLayer(nn.Module):
  def __init__(
    self,
    hidden_dim,
    n_heads,
    pre_query_mode,
    query_process_mode,
    model_length,
    normalize_pre_queries,
  ):
    super().__init__()
    assert hidden_dim % n_heads == 0
    self.hidden_dim = hidden_dim
    self.n_heads = n_heads
    self.pre_query_mode = pre_query_mode
    self.query_process_mode = query_process_mode
    self.model_length = model_length
    self.normalize_pre_queries = normalize_pre_queries

    self._prepare_query_processing()
    self.hidden_to_key_value = nn.Linear(hidden_dim, 2 * hidden_dim)
    self.output_linear = nn.Linear(hidden_dim, hidden_dim)

  def _prepare_query_processing(self):
    if self.pre_query_mode not in {'learn', 'learn+freqs', 
                                   'learn+freqs+mean', 
                                   'learn+freqs+logsumexp'}:
      raise ValueError(self.pre_query_mode)

    if self.query_process_mode not in {'linear', 'mlp4'}:
      raise ValueError(self.pre_query_mode)

    if self.normalize_pre_queries == 'layernorm':
      self.pre_query_norm = nn.LayerNorm(self.hidden_dim)
    elif self.normalize_pre_queries == 'rmsnorm':
      self.pre_query_mode = nn.RMSNorm(self.hidden_dim)
    else:
      raise ValueError(self.normalize_pre_queries)

    self.base_embedding = nn.Embedding(1, 
      self.hidden_dim).weight

    if 'freqs' in self.pre_query_mode:
      seq_pos = torch.arange(self.model_length)
      pos_freqs = get_sinusoidal_embedding(seq_pos, 
                                           self.hidden_dim)
      self.register_buffer('pos_freqs', pos_freqs, 
                            persistent=False)

    ### SECOND, PREPARE THE PRE-QUERY PROCESSING
    if self.query_process_mode == 'linear':
      self.query_processor = nn.Linear(self.hidden_dim, 
                                       self.hidden_dim)
    elif self.query_process_mode == 'mlp4':
      self.query_processor = nn.Sequential(
      nn.Linear(self.hidden_dim, 4 * self.hidden_dim),
      nn.SiLU(),
      nn.Linear(4 * self.hidden_dim, self.hidden_dim))
    else:
      raise ValueError(self.query_process_mode)
    
  def _compute_queries(self, x, positions, group_idxs, 
                       concrete_lengths, use_inference_mode):
    queries = self.base_embedding[None]  # 1 x 1 x H
    if 'freqs' in self.pre_query_mode:
      if not use_inference_mode:
        freqs = self.pos_freqs[None]  # 1 x L x H
        freqs = self.pos_freqs[None, :x.shape[1]]
      else:
        freqs = _index_freqs_swap(self.pos_freqs, positions)
      queries = queries + freqs

    if self.pre_query_mode.startswith('learn+freqs+'):
      # learn+freqs+mean, learn+freqs+logsumexp
      if not use_inference_mode:
        # Training: need to compute a value per group
        extract_fn = (_partition_logsumexp_train 
                      if 'logsumexp' in self.pre_query_mode 
                      else _partition_mean_train)
        _, val_grp_zero = extract_fn(x, group_idxs, 0)
        mask_one, val_grp_one = extract_fn(x, group_idxs, 1)
        queries = queries + torch.where(mask_one[..., None],  # (bs, l, dim)
                                        val_grp_zero[:, None],  
                                        val_grp_one[:, None])
      else:
        # Inference: all tokens are in the same group -> just broadcast
        extract_fn = (_partition_logsumexp_inference
                      if 'logsumexp' in self.pre_query_mode
                      else _partition_mean_inference)
        value = extract_fn(x, concrete_lengths)  # bs x dim
        queries = queries + value[:, None, :]

    queries = self.pre_query_norm(queries)
    out = self.query_processor(queries)
    return out
  
  def forward(
    self,
    x,
    rotary_cos_sin_queries,
    rotary_cos_sin_keys,
    # Training
    group_idxs,
    # Inference
    position_queries,
    concrete_lengths,
    cross_attn_mask,
    use_inference_mode):
    group_queries = self._compute_queries(x, position_queries, 
                                          group_idxs, 
                                          concrete_lengths,
                                          use_inference_mode)
    keys, values = self.hidden_to_key_value(x).split(self.hidden_dim, 
                                                     dim=-1)
    pattern = 'bs l (n_heads head_dim) -> bs l n_heads head_dim'
    group_queries = einops.rearrange(group_queries, pattern, n_heads=self.n_heads)
    keys = einops.rearrange(keys, pattern, n_heads=self.n_heads)

    if self.pre_query_mode == 'learn':
      # When using learn pre_query_mode, the queries are shared for all positions
      #  but we need to repeat it to be able to apply RoPE
      if use_inference_mode:
        # Expand to number of queires
        expand_value = position_queries.shape[1]
      else:  # Training
        # Expand to number of keys (keys/queries have the same shape)
        expand_value = keys.shape[1]
      group_queries = group_queries.repeat(1, expand_value, 
                                           1, 1)
      
    if self.pre_query_mode in ('learn', 'learn+freqs') \
                                and not use_inference_mode:
      group_queries = group_queries.repeat(keys.shape[0], 1, 
                                           1, 1)
    group_queries = apply_rotary_pos_emb_single(group_queries, 
                                                *rotary_cos_sin_queries)
    keys = apply_rotary_pos_emb_single(keys, *rotary_cos_sin_keys)
    values = einops.rearrange(values, 
      'bs l (n_heads head_dim) -> bs l n_heads head_dim', 
      n_heads=self.n_heads)

    out = sdpa_attention_masked(group_queries, keys, values, cross_attn_mask, causal=False)
    out = self.output_linear(out)
    return out


class Decoder(nn.Module):
  def __init__(self, n_blocks, dim, n_heads, cond_dim, 
               mlp_ratio, dropout, adaLN, model_length, 
               swap_pre_query_mode, swap_query_process_mode, 
               swap_normalize_mode):
    super().__init__()

    self.hidden_dim = dim
    self.n_heads = n_heads
    self.cond_dim = cond_dim
    self.mlp_ratio = mlp_ratio
    self.adaLN = adaLN
    self.model_length = model_length
    self.dropout = dropout
    self.n_blocks = n_blocks
    self.swap_pre_query_mode = swap_pre_query_mode
    self.group_swap = GroupSwapLayer(dim, n_heads, 
                swap_pre_query_mode, swap_query_process_mode, 
                model_length, swap_normalize_mode)

    self.layers = nn.ModuleList([self._make_cross_attn_block() 
                                for _ in range(self.n_blocks)])

  def _make_cross_attn_block(self):
    return CrossAttnDDiTBlock(
      self.hidden_dim, self.n_heads, self.adaLN, self.cond_dim, 
      self.mlp_ratio, self.dropout)

  def forward(
    self, 
    encoder_output,
    t_cond, 
    rotary_cos_sin_queries, 
    rotary_cos_sin_keys, 
    self_attn_mask,
    # Training
    group_idxs, 
    # Inference
    position_queries,
    concrete_lengths_keys, 
    use_inference_mode,
  ):
    """
    1. Apply GroupSwap -> prepare cross attention mask
    2. Apply layers
    """
    if not use_inference_mode:  # Training / Valid
      cross_attn_mask = make_group_cross_attn_mask(group_idxs)
      q_len = self.model_length
    else:  # Sampling
      q_len = position_queries.shape[1]
      kv_len = encoder_output.shape[1]
      cross_attn_mask = make_inference_cross_attn_mask(
        kv_len, q_len, concrete_lengths_keys)
      # IMPORTANT NOTE: during inference, the self attention 
      #  mask is different than during training, since the
      #  decoder input has a different shape than the encoder
      #  input.
      del self_attn_mask  # will not be used

    x = self.group_swap(encoder_output, rotary_cos_sin_queries,
      rotary_cos_sin_keys, group_idxs, position_queries, 
      concrete_lengths_keys, cross_attn_mask, use_inference_mode)

    for layer in self.layers:
      if isinstance(layer, DDiTBlock):  # self attention
        x = layer(x, t_cond, rotary_cos_sin_queries, 
                  self_attn_mask)
      else:  # cross attention
        x = layer(
          q_x=x,
          kv_x=encoder_output,
          t_cond=t_cond, 
          rotary_cos_sin_queries=rotary_cos_sin_queries,
          rotary_cos_sin_keys=rotary_cos_sin_keys,
          attn_mask=cross_attn_mask)
    return x
  

class PartitionDIT(nn.Module):
  def __init__(self, config, vocab_size: int):
    super().__init__()
    if type(config) == dict:
      config = omegaconf.OmegaConf.create(config)

    assert not config.algo.causal_attention
    self.adaLN = True
    self.config = config
    self.vocab_size = vocab_size
    dim = config.model.hidden_size
    self.vocab_embed = EmbeddingLayer(dim, vocab_size)
    self.rotary_emb = Rotary(dim // config.model.n_heads)
    self.sigma_map = TimestepEmbedder(config.model.cond_dim)
    self.n_heads = config.model.n_heads
    self.model_length = config.model.length
    self.encoder = Encoder(
      config.model.encoder.n_blocks,
      dim,
      config.model.n_heads,
      config.model.cond_dim,
      config.model.mlp_ratio,
      config.model.dropout,
      self.adaLN)
    self.decoder = Decoder(
      config.model.decoder.n_blocks,
      dim,
      config.model.n_heads,
      config.model.cond_dim,
      config.model.mlp_ratio,
      config.model.dropout,
      self.adaLN,
      config.model.length,
      config.model.swap.pre_query_mode,
      config.model.swap.query_process_mode,
      config.model.swap.normalize_mode)
    self.output_layer = DDiTFinalLayer(dim, vocab_size,
                                       config.model.cond_dim, 
                                       self.adaLN)

  def forward(
    self, 
    x, 
    sigma, 
    # Training
    group_idxs=None, 
    # Inference
    clean_positions=None, 
    noisy_positions=None, 
    concrete_lengths=None,
    use_inference_mode=False,
  ):
    x = self.vocab_embed(x)
    t_cond = F.silu(self.sigma_map(sigma))
    rotary_cos_sin = self.rotary_emb(seq_len=self.model_length, 
                                     device=x.device)
    if not use_inference_mode:  # Training
      assert group_idxs is not None
      assert clean_positions is None
      assert noisy_positions is None
      assert concrete_lengths is None
      self_attn_mask = make_group_self_attn_mask(group_idxs)
      rotary_cos_sin_queries = rotary_cos_sin
      rotary_cos_sin_keys = rotary_cos_sin
    else:  # Inference. NOTE: the self-attn mask is only for 
      #      the encoder during inference mode!!!
      assert group_idxs is None
      assert clean_positions is not None
      assert noisy_positions is not None
      assert concrete_lengths is not None
      self_attn_mask = make_inference_self_attn_mask(
        x.shape[1], concrete_lengths)
      rotary_cos_sin_queries = tuple(_index_rotary(vec, noisy_positions) 
                                     for vec in rotary_cos_sin)
      rotary_cos_sin_keys = tuple(_index_rotary(vec, clean_positions) 
                                  for vec in rotary_cos_sin)

    with torch.amp.autocast('cuda', dtype=torch.bfloat16):
      enc_out = self.encoder(x, t_cond, rotary_cos_sin_keys, 
                             self_attn_mask)
      dec_out = self.decoder(enc_out, t_cond, 
        rotary_cos_sin_queries, rotary_cos_sin_keys, 
        self_attn_mask, group_idxs, noisy_positions, 
        concrete_lengths, use_inference_mode)
      out = self.output_layer(dec_out, t_cond)
    return out
