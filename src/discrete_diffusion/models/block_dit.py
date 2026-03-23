"""Block-diffusion aware DiT backbone."""

from __future__ import annotations

import math
import typing
from functools import partial

import einops
from einops import rearrange
import huggingface_hub
import omegaconf
import torch
import torch.nn as nn
import torch.nn.functional as F
from .common import (
  bias_dropout_add_scale,
  get_bias_dropout_add_scale,
  bias_dropout_add_scale_fused_train,
  bias_dropout_add_scale_fused_inference,
  modulate,
  modulate_fused,
  LayerNorm,
  Rotary,
  rotate_half,
  split_and_apply_rotary_pos_emb,
  apply_rotary_pos_emb,
  apply_rotary_pos_emb_torchscript,
  TimestepEmbedder,
  EmbeddingLayer,
  DDiTFinalLayer,
  sdpa_attention_unmasked,
  sdpa_attention_masked,
  flash_varlen_attention_qkvpacked,
)

try:  # flex attention is optional
  from torch.nn.attention.flex_attention import flex_attention, create_block_mask

  FLEX_ATTN_AVAILABLE = True
except (ImportError, RuntimeError):
  flex_attention = None  # type: ignore
  create_block_mask = None  # type: ignore
  FLEX_ATTN_AVAILABLE = False

try:  # flash-attn is optional but recommended
  import flash_attn
  import flash_attn.layers.rotary
except (ImportError, RuntimeError):  # pragma: no cover - optional dependency
  flash_attn = None  # type: ignore

# Torch jit fusion flags, mirrored from upstream BD3-LM implementation
torch._C._jit_set_profiling_mode(False)
torch._C._jit_set_profiling_executor(False)
torch._C._jit_override_can_fuse_on_cpu(True)
torch._C._jit_override_can_fuse_on_gpu(True)


def block_diff_mask(b, h, q_idx, kv_idx, block_size=None, n=None):  # noqa: D401, ignored args
  """Construct the block diffusion attention mask.

  Line-by-line match to upstream BD3-LM implementation.
  """

  x0_flag_q = (q_idx >= n)
  x0_flag_kv = (kv_idx >= n)

  block_q = torch.where(
    x0_flag_q == 1,
    (q_idx - n) // block_size,
    q_idx // block_size)
  block_kv = torch.where(
    x0_flag_kv == 1,
    (kv_idx - n) // block_size,
    kv_idx // block_size)

  block_diagonal = (block_q == block_kv) & (x0_flag_q == x0_flag_kv)
  offset_block_causal = (
    (block_q > block_kv)
    & (x0_flag_kv == 1)
    & (x0_flag_q == 0))
  block_causal = (
    (block_q >= block_kv)
    & (x0_flag_kv == 1)
    & (x0_flag_q == 1))
  return block_diagonal | offset_block_causal | block_causal


@torch.compile(fullgraph=True, mode="max-autotune-no-cudagraphs")  # type: ignore[misc]
def fused_flex_attention(q, k, v, mask=None):  # pragma: no cover - requires flex attention runtime
  return flex_attention(q, k, v, block_mask=mask)




class DDiTBlockCausal(nn.Module):
  def __init__(self, n, dim, n_heads, mlp_ratio=4, dropout=0.1,
               max_batch_size=64, max_seqlen=1024, adaLN=False,
               cond_dim=None, attn_backend='auto'):
    super().__init__()
    self.n_heads = n_heads
    self.max_seqlen = max_seqlen
    self.n = n

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
    self.adaLN = adaLN
    if self.adaLN:
      self.adaLN_modulation = nn.Linear(cond_dim, 6 * dim)
      self.adaLN_modulation.weight.data.zero_()
      self.adaLN_modulation.bias.data.zero_()
    self.attn_backend = attn_backend
    self.kv_cache = None

  def _get_bias_dropout_scale(self):
    if self.training:
      return bias_dropout_add_scale_fused_train
    return bias_dropout_add_scale_fused_inference

  def get_qkv(self, x, rotary_cos_sin, store_kv=False):
    if self.kv_cache is not None:
      new_qkv = self.attn_qkv(x[:, -1:])
      qkv = torch.cat((self.kv_cache, new_qkv), dim=1)
    else:
      qkv = self.attn_qkv(x)
    if store_kv:
      self.kv_cache = qkv[:, -(self.max_seqlen - 1):].clone()

    qkv = einops.rearrange(
      qkv,
      'b s (three h d) -> b s three h d',
      three=3,
      h=self.n_heads)
    with torch.amp.autocast('cuda', enabled=False):
      cos, sin = rotary_cos_sin
      if self.attn_backend == 'flash_attn':
        qkv = apply_rotary_pos_emb(
          qkv, cos.to(qkv.dtype), sin.to(qkv.dtype))
      else:
        qkv = apply_rotary_pos_emb_torchscript(
          qkv, cos.to(qkv.dtype), sin.to(qkv.dtype))
    return qkv

  def cross_attn(self, qkv, mask=None):
    scale = qkv.shape[-1]
    qkv = qkv.transpose(1, 3)
    mask = mask.bool() if mask is not None else None
    x = F.scaled_dot_product_attention(
      query=qkv[:, :, 0],
      key=qkv[:, :, 1],
      value=qkv[:, :, 2],
      attn_mask=mask,
      is_causal=True,
      scale=1 / math.sqrt(scale))
    x = x.transpose(1, 2)
    x = rearrange(x, 'b s h d -> b s (h d)')
    return x

  def forward(self,
              x,
              rotary_cos_sin,
              c=None,
              causal=True,
              mask=None,
              store_kv=False,
              **kwargs):
    del kwargs
    batch_size, seq_len = x.shape[0], x.shape[1]
    shift_msa = scale_msa = gate_msa = None
    shift_mlp = scale_mlp = gate_mlp = None
    bias_dropout_scale_fn = self._get_bias_dropout_scale()
    if c is not None and c.shape[0] == batch_size:
      (shift_msa, scale_msa, gate_msa, shift_mlp,
       scale_mlp, gate_mlp) = self.adaLN_modulation(c)[:, None].chunk(6, dim=2)
    elif c is not None:
      (shift_msa, scale_msa, gate_msa, shift_mlp,
       scale_mlp, gate_mlp) = rearrange(
         self.adaLN_modulation(c), '(b h) d -> b h d', b=batch_size
         ).chunk(6, dim=-1)

    x_skip = x
    if c is not None:
      x = modulate_fused(self.norm1(x), shift_msa, scale_msa)
    else:
      x = self.norm1(x)

    qkv = self.get_qkv(x, rotary_cos_sin, store_kv=store_kv)
    if self.attn_backend == 'flash_attn':
      assert flash_attn is not None
      x = flash_varlen_attention_qkvpacked(qkv, causal=True)
    else:
      x = self.cross_attn(qkv, c)

    scale = torch.ones(1, device=x.device, dtype=x.dtype)
    if c is not None:
      x = bias_dropout_scale_fn(self.attn_out(x), None, gate_msa, x_skip, self.dropout)
      x = bias_dropout_scale_fn(
        self.mlp(modulate_fused(
          self.norm2(x), shift_mlp, scale_mlp)),
        None, gate_mlp, x, self.dropout)
    else:
      x = bias_dropout_scale_fn(
        self.attn_out(x), None, scale, x_skip, self.dropout)
      x = bias_dropout_scale_fn(
        self.mlp(self.norm2(x)), None, scale, x, self.dropout)
    return x


class DDiTBlock(nn.Module):
  def __init__(self, n, dim, n_heads, adaLN,
               latent_dim=None, cond_dim=None,
               latent_conditioning=-1, mlp_ratio=4,
               dropout=0.1, block_size=1,
               max_batch_size=64, max_seqlen=1024, attn_backend='auto'):
    super().__init__()
    self.max_seqlen = max_seqlen
    self.n = n
    self.n_heads = n_heads
    self.adaLN = adaLN
    self.latent_conditioning = latent_conditioning
    self.block_size = block_size

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
    self.kv_cache = None
    self.cache_idx = 0

    if self.adaLN:
      self.adaLN_modulation = nn.Linear(cond_dim, 6 * dim)
      self.adaLN_modulation.weight.data.zero_()
      self.adaLN_modulation.bias.data.zero_()
    self.attn_backend = attn_backend

  def _get_bias_dropout_scale(self):
    if self.training:
      return bias_dropout_add_scale_fused_train
    return bias_dropout_add_scale_fused_inference

  def get_qkv(self, x, rotary_cos_sin, store_kv=False):
    if self.kv_cache is not None:
      new_qkv = self.attn_qkv(x)
      self.kv_cache[:, self.cache_idx:self.cache_idx + self.block_size] = new_qkv
      qkv = self.kv_cache[:, :self.cache_idx + self.block_size].clone()
    else:
      qkv = self.attn_qkv(x)
    if store_kv:
      self.cache_idx += self.block_size
      if self.cache_idx >= self.max_seqlen:
        self.cache_idx = self.max_seqlen - self.block_size
        self.kv_cache[:, :-self.block_size] = self.kv_cache[:, self.block_size:].clone()

    qkv = einops.rearrange(
      qkv,
      'b s (three h d) -> b s three h d',
      three=3,
      h=self.n_heads)
    with torch.amp.autocast('cuda', enabled=False):
      cos, sin = rotary_cos_sin
      if self.attn_backend == 'flash_attn':
        qkv = apply_rotary_pos_emb(
          qkv, cos.to(qkv.dtype), sin.to(qkv.dtype))
      else:
        qkv = apply_rotary_pos_emb_torchscript(
          qkv, cos.to(qkv.dtype), sin.to(qkv.dtype))
    return qkv

  def attn_mlp(self, x, c, gate_msa, gate_mlp, shift_mlp, scale_mlp, x_skip):
    bias_dropout_scale_fn = self._get_bias_dropout_scale()
    if c is not None:
      x = bias_dropout_scale_fn(self.attn_out(x), None, gate_msa, x_skip, self.dropout)
      x = bias_dropout_scale_fn(
        self.mlp(modulate_fused(self.norm2(x), shift_mlp, scale_mlp)),
        None, gate_mlp, x, self.dropout)
    else:
      scale = torch.ones(1, device=x.device, dtype=x.dtype)
      x = bias_dropout_scale_fn(
        self.attn_out(x), None, scale, x_skip, self.dropout)
      x = bias_dropout_scale_fn(
        self.mlp(self.norm2(x)), None, scale, x, self.dropout)
    return x

  def cross_attn(self, qkv, mask=None):
    scale = qkv.shape[-1]
    qkv = qkv.transpose(1, 3)
    mask = mask.bool() if mask is not None else None
    x = F.scaled_dot_product_attention(
      query=qkv[:, :, 0],
      key=qkv[:, :, 1],
      value=qkv[:, :, 2],
      attn_mask=mask,
      is_causal=False,
      scale=1 / math.sqrt(scale))
    x = x.transpose(1, 2)
    x = rearrange(x, 'b s h d -> b s (h d)')
    return x

  def cross_attn_flex(self, qkv, mask=None):  # pragma: no cover - requires flex attention runtime
    qkv = rearrange(qkv, 'b s three h d -> b h three s d', h=self.n_heads)
    x = fused_flex_attention(qkv[:, :, 0], qkv[:, :, 1], qkv[:, :, 2], mask=mask)
    x = rearrange(x, 'b h s d -> b s (h d)')
    return x

  def forward(self,
              x,
              rotary_cos_sin,
              c,
              causal=False,
              mask=None,
              sample_mode=False,
              store_kv=False):
    batch_size, seq_len = x.shape[0], x.shape[1]

    shift_msa = scale_msa = gate_msa = None
    shift_mlp = scale_mlp = gate_mlp = None
    if c is not None and c.shape[0] == batch_size:
      (shift_msa, scale_msa, gate_msa, shift_mlp,
       scale_mlp, gate_mlp) = self.adaLN_modulation(c)[:, None].chunk(6, dim=2)
    elif c is not None:
      (shift_msa, scale_msa, gate_msa, shift_mlp,
       scale_mlp, gate_mlp) = rearrange(
         self.adaLN_modulation(c), '(b h) d -> b h d', b=batch_size
         ).chunk(6, dim=-1)

    x_skip = x
    if c is not None:
      x = modulate_fused(self.norm1(x), shift_msa, scale_msa)
    else:
      x = self.norm1(x)

    if mask is not None and not sample_mode:
      qkv_x = self.get_qkv(x[:, :self.n], rotary_cos_sin)
      qkv_x0 = self.get_qkv(x[:, self.n:], rotary_cos_sin)
      qkv = torch.cat((qkv_x, qkv_x0), dim=1)
    else:
      qkv = self.get_qkv(x, rotary_cos_sin, store_kv=store_kv)

    if (self.attn_backend == 'flash_attn' or (self.attn_backend == 'auto' and flash_attn is not None)) and mask is None:
      assert flash_attn is not None
      x = flash_varlen_attention_qkvpacked(qkv, causal=causal)
    elif (self.attn_backend == 'flex' or (self.attn_backend == 'auto' and FLEX_ATTN_AVAILABLE)) and FLEX_ATTN_AVAILABLE:
      x = self.cross_attn_flex(qkv, mask=mask)
    elif self.attn_backend in {'sdpa', 'auto', 'flex'}:
      x = self.cross_attn(qkv, mask=mask)
    else:
      raise ValueError('Unknown attention backend')
    if self.kv_cache is not None:
      x = x[:, -self.block_size:]
    x = self.attn_mlp(x, c, gate_msa, gate_mlp, shift_mlp, scale_mlp, x_skip)
    return x


class BlockDiT(nn.Module, huggingface_hub.PyTorchModelHubMixin):
  """DiT backbone extended with block-diffusion attention masks."""

  def __init__(self, config, vocab_size: int):
    super().__init__()
    if isinstance(config, dict):
      config = omegaconf.OmegaConf.create(config)
    self.config = config
    self.n = config.model.length
    self.causal = getattr(config.model, 'causal_attention', config.algo.parameterization == 'ar')
    self.adaLN = (not self.causal) or getattr(config.model, 'adaln', False)
    self.vocab_size = vocab_size
    self.block_size = getattr(config, 'block_size', config.model.length)
    dim = config.model.hidden_size
    cond_dim = config.model.cond_dim
    self.n_heads = config.model.n_heads
    self.vocab_embed = EmbeddingLayer(dim, vocab_size)
    if self.adaLN or not self.causal:
      self.sigma_map = TimestepEmbedder(cond_dim)
    self.rotary_emb = Rotary(dim // config.model.n_heads)
    self.attn_backend = getattr(config.model, 'attn_backend', 'auto')
    self.max_seqlen = 1024

    blocks = []
    for _ in range(config.model.n_blocks):
      if self.causal:
        block = DDiTBlockCausal(
          n=config.model.length,
          dim=dim,
          n_heads=config.model.n_heads,
          dropout=config.model.dropout,
          max_batch_size=config.loader.eval_batch_size,
          adaLN=self.adaLN,
          cond_dim=cond_dim,
          attn_backend=self.attn_backend)
      else:
        block = DDiTBlock(
          n=config.model.length,
          dim=dim,
          n_heads=config.model.n_heads,
          cond_dim=cond_dim,
          adaLN=self.adaLN,
          dropout=config.model.dropout,
          block_size=self.block_size,
          attn_backend=self.attn_backend,
          max_seqlen=self.max_seqlen)
      blocks.append(block)
    self.blocks = nn.ModuleList(blocks)
    self.output_layer = DDiTFinalLayer(
      hidden_size=dim,
      out_channels=vocab_size,
      cond_dim=cond_dim,
      adaLN=self.adaLN,
      tie_word_embeddings=getattr(config.model, 'tie_word_embeddings', False))
    # Tie output projection to input embeddings if requested
    if getattr(config.model, 'tie_word_embeddings', False):
      self.output_layer.linear.weight = self.vocab_embed.embedding
    if getattr(config.algo, 'cross_attn', False):
      self.gen_mask(config.model.length, self.block_size, self.attn_backend)

  def _get_bias_dropout_scale(self):
    if self.training:
      return bias_dropout_add_scale_fused_train
    return bias_dropout_add_scale_fused_inference

  def gen_mask(self, seqlen, block_size, attn_backend='sdpa'):
    if attn_backend == 'flex' and FLEX_ATTN_AVAILABLE:
      assert create_block_mask is not None
      self.block_diff_mask = create_block_mask(
        partial(block_diff_mask, block_size=block_size, n=seqlen),
        B=None, H=None, Q_LEN=seqlen * 2, KV_LEN=seqlen * 2)
    elif attn_backend == 'sdpa':
      self.block_diff_mask = block_diff_mask(
        b=None, h=None, q_idx=torch.arange(seqlen * 2)[:, None],
        kv_idx=torch.arange(seqlen * 2)[None, :],
        block_size=block_size, n=seqlen)
    else:
      raise ValueError('Unknown attention backend')

  def reset_kv_cache(self):
    for block in self.blocks:
      block.kv_cache = torch.zeros(
        self.config.loader.eval_batch_size,
        self.max_seqlen,
        self.config.model.hidden_size * 3,
        device='cuda',
        dtype=torch.bfloat16)
      block.cache_idx = 0

  def forward(self, indices, sigma, sample_mode=False, store_kv=False):
    x = self.vocab_embed(indices)
    if sigma is None:
      t_cond = None
    else:
      t_cond = F.silu(self.sigma_map(sigma))

    cross_attn = hasattr(self, 'block_diff_mask')
    if cross_attn:
      mask = self.block_diff_mask
      if sample_mode:
        if getattr(self.config.sampling, 'kv_cache', False):
          mask = None
          accum_length = self.blocks[0].cache_idx + self.block_size
          x_full = torch.zeros((
            x.shape[0], accum_length, x.shape[2]), device=x.device)
          rotary_cos_sin = self.rotary_emb(x_full)
        else:
          mask = mask[
            self.n:self.n + x.shape[1], self.n:self.n + x.shape[1]]
          rotary_cos_sin = self.rotary_emb(x)
      else:
        rotary_cos_sin = self.rotary_emb(x[:, :self.n])
    else:
      rotary_cos_sin = self.rotary_emb(x)
      mask = None

    with torch.amp.autocast('cuda', dtype=torch.bfloat16):
      for block in self.blocks:
        x = block(
          x,
          rotary_cos_sin,
          c=t_cond,
          causal=self.causal,
          sample_mode=sample_mode,
          mask=mask,
          store_kv=store_kv)
      x = self.output_layer(x, t_cond)
    if cross_attn and not sample_mode:
      x = x[:, :self.n]
    return x


__all__ = ['BlockDiT']
