"""Campbell model."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
import einops
import omegaconf

from .common import (
    EmbeddingLayer,
    TimestepEmbedder,
    LayerNorm,
    Rotary,
    DDiTFinalLayer,
    apply_rotary_pos_emb_single,
    sdpa_attention_masked,
    modulate_fused,
    bias_dropout_add_scale_fused_train,
    bias_dropout_add_scale_fused_inference,
)
from ..utils.rank_masks import (
    compute_rank_from_tau,
    normalize_attn_mask,
)


def _pairwise_rank(rank: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    return rank[:, :, None], rank[:, None, :]


def _apply_pad_constraints(mask: torch.Tensor, pad_mask: torch.Tensor | None) -> torch.Tensor:
    if pad_mask is None:
        return mask
    valid_q = ~pad_mask[:, :, None]
    valid_k = ~pad_mask[:, None, :]
    return mask & valid_q & valid_k

def build_M_le(rank: torch.Tensor, pad_mask: torch.Tensor | None = None) -> torch.Tensor:
    """Rank-causal self-attention mask."""
    ri, rj = _pairwise_rank(rank)
    return _apply_pad_constraints(rj <= ri, pad_mask)


def build_M_ge(rank: torch.Tensor, pad_mask: torch.Tensor | None = None) -> torch.Tensor:
    """Reverse-rank self-attention mask."""
    ri, rj = _pairwise_rank(rank)
    return _apply_pad_constraints(rj >= ri, pad_mask)


def build_M_QQ(rank: torch.Tensor, pad_mask: torch.Tensor | None = None) -> torch.Tensor:
    """Query self-attention mask."""
    return build_M_le(rank, pad_mask)


def build_M_hybrid(rank: torch.Tensor, pad_mask: torch.Tensor | None = None) -> torch.Tensor:
    """Cross-attention mask from query to clean and observed memory."""
    ri, rj = _pairwise_rank(rank)
    allow_c = _apply_pad_constraints(rj < ri, pad_mask)
    allow_o = _apply_pad_constraints(rj >= ri, pad_mask)
    return torch.cat([allow_c, allow_o], dim=2)


# =============================================================================
# Safety Assertions (Debug Mode)
# =============================================================================

def assert_hybrid_mask_safety(M_hybrid: torch.Tensor, rank: torch.Tensor):
    """Verify the strict clean-memory boundary used by the hybrid mask."""
    B, n, _ = M_hybrid.shape

    C_part = M_hybrid[:, :, :n]
    C_diag = torch.diagonal(C_part, dim1=1, dim2=2)
    assert not C_diag.any(), "Hybrid mask allows query access to clean self tokens."

    rank_0_mask = rank == 0
    O_part = M_hybrid[:, :, n:]

    for b in range(B):
        rank_0_positions = torch.where(rank_0_mask[b])[0]
        for pos in rank_0_positions:
            C_valid = C_part[b, pos].any().item()
            O_valid = O_part[b, pos].any().item()
            assert not C_valid, f"Rank-0 position {pos} has invalid clean-memory access."
            assert O_valid, f"Rank-0 position {pos} has no observed-memory access."


# =============================================================================
# Network Components
# =============================================================================

class MemoryBlock(nn.Module):
    """Shared memory encoder block for Content and Observation streams.
    
    This is a generic transformer block with self-attention and MLP.
    The same weights are used for both C and O streams with different masks:
    - C stream uses M_le (rank-causal, includes self)
    - O stream uses M_ge (reverse-rank, includes self)
    
    Time conditioning is configurable.
    """
    
    def __init__(
        self,
        dim: int,
        n_heads: int,
        cond_dim: int | None = None,
        mlp_ratio: int = 4,
        dropout: float = 0.1,
        use_time: bool = False,
    ):
        super().__init__()
        self.dim = dim
        self.n_heads = n_heads
        self.head_dim = dim // n_heads
        self.use_time = use_time
        self.dropout = dropout
        
        self.norm1 = LayerNorm(dim)
        self.attn_qkv = nn.Linear(dim, 3 * dim, bias=False)
        self.attn_out = nn.Linear(dim, dim, bias=False)
        
        self.norm2 = LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_ratio * dim, bias=True),
            nn.GELU(approximate='tanh'),
            nn.Linear(mlp_ratio * dim, dim, bias=True),
        )
        
        if use_time and cond_dim is not None:
            self.adaLN_modulation = nn.Linear(cond_dim, 6 * dim)
            self.adaLN_modulation.weight.data.zero_()
            self.adaLN_modulation.bias.data.zero_()
        else:
            self.adaLN_modulation = None
    
    def forward(
        self,
        x: torch.Tensor,
        t_cond: torch.Tensor | None,
        rotary_cos_sin: Tuple[torch.Tensor, torch.Tensor],
        attn_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Forward pass with self-attention.
        
        Args:
            x: [B, n, D] input hidden states
            t_cond: [B, n, cond_dim] time conditioning (or None)
            rotary_cos_sin: RoPE embeddings
            attn_mask: [B, n, n] additive attention mask
            
        Returns:
            Updated hidden states [B, n, D]
        """
        bias_dropout_scale_fn = (
            bias_dropout_add_scale_fused_train if self.training
            else bias_dropout_add_scale_fused_inference
        )
        
        x_before = x
        x_normed = self.norm1(x)
        
        shift_msa = scale_msa = gate_msa = None
        shift_mlp = scale_mlp = gate_mlp = None
        
        if self.use_time and self.adaLN_modulation is not None and t_cond is not None:
            t_cond_mod = self.adaLN_modulation(t_cond)
            (shift_msa, scale_msa, gate_msa,
             shift_mlp, scale_mlp, gate_mlp) = t_cond_mod.chunk(6, dim=2)
            x_normed = modulate_fused(x_normed, shift_msa, scale_msa)
        
        qkv = einops.rearrange(
            self.attn_qkv(x_normed),
            'b s (three h d) -> b s three h d',
            three=3, h=self.n_heads
        )
        
        cos, sin = rotary_cos_sin
        cos = cos.to(qkv.dtype)
        sin = sin.to(qkv.dtype)
        
        q = qkv[:, :, 0]
        k = qkv[:, :, 1]
        v = qkv[:, :, 2]
        
        q = apply_rotary_pos_emb_single(q, cos, sin)
        k = apply_rotary_pos_emb_single(k, cos, sin)
        
        x_attn = sdpa_attention_masked(q, k, v, attn_mask, causal=False)
        
        if gate_msa is not None:
            x = bias_dropout_scale_fn(
                self.attn_out(x_attn), None, gate_msa, x_before, self.dropout
            )
            x_before_mlp = x
            x_normed_mlp = self.norm2(x)
            x_normed_mlp = modulate_fused(x_normed_mlp, shift_mlp, scale_mlp)
            x = bias_dropout_scale_fn(
                self.mlp(x_normed_mlp), None, gate_mlp, x_before_mlp, self.dropout
            )
        else:
            scale = torch.ones(1, device=x_attn.device, dtype=x_attn.dtype)
            x = bias_dropout_scale_fn(
                self.attn_out(x_attn), None, scale, x_before, self.dropout
            )
            x = bias_dropout_scale_fn(
                self.mlp(self.norm2(x)), None, scale, x, self.dropout
            )
        
        return x


class CampbellQueryBlock(nn.Module):
    """Query block with self-attention and hybrid cross-attention."""
    
    def __init__(
        self,
        dim: int,
        n_heads: int,
        cond_dim: int,
        mlp_ratio: int = 4,
        dropout: float = 0.1,
        use_time: bool = True,
    ):
        super().__init__()
        self.dim = dim
        self.n_heads = n_heads
        self.head_dim = dim // n_heads
        self.use_time = use_time
        self.dropout = dropout
        
        # =================================================================
        # Q-Q Self-Attention components
        # =================================================================
        self.q_self_norm = LayerNorm(dim)
        self.q_self_qkv = nn.Linear(dim, 3 * dim, bias=False)
        self.q_self_out = nn.Linear(dim, dim, bias=False)
        
        # =================================================================
        # Q -> [C||O] Cross-Attention components
        # =================================================================
        self.q_cross_norm = LayerNorm(dim)
        self.mem_norm = LayerNorm(dim)
        
        self.q_proj = nn.Linear(dim, dim, bias=False)
        self.kv_proj = nn.Linear(dim, 2 * dim, bias=False)
        self.cross_out = nn.Linear(dim, dim, bias=False)
        
        # =================================================================
        # MLP
        # =================================================================
        self.norm_mlp = LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_ratio * dim, bias=True),
            nn.GELU(approximate='tanh'),
            nn.Linear(mlp_ratio * dim, dim, bias=True),
        )
        
        # =================================================================
        # AdaLN modulation (for self-attn, cross-attn, and MLP)
        # =================================================================
        if use_time:
            # 6 params for self-attn + 6 for cross-attn + 6 for mlp = 18 total
            # But we'll use simpler structure: 6 for combined attention + 6 for mlp
            self.adaLN_modulation = nn.Linear(cond_dim, 12 * dim)
            self.adaLN_modulation.weight.data.zero_()
            b = self.adaLN_modulation.bias.data
            b.zero_()
            # Initialize gates to 0.5 for stability
            b[2*dim:3*dim] = 0.5  # gate for self-attn
            b[5*dim:6*dim] = 0.5  # gate for cross-attn
            b[8*dim:9*dim] = 0.5  # gate for mlp (first set)
            b[11*dim:12*dim] = 0.5  # gate for mlp (second set)
        else:
            self.adaLN_modulation = None
    
    def forward(
        self,
        q_x: torch.Tensor,
        h_content: torch.Tensor,
        h_obs: torch.Tensor,
        t_cond: torch.Tensor | None,
        rotary_cos_sin: Tuple[torch.Tensor, torch.Tensor],
        M_QQ_additive: torch.Tensor,  # [B, n, n] Q-Q self-attn mask
        M_hybrid_additive: torch.Tensor,  # [B, n, 2n] Q -> [C||O] cross-attn mask
    ) -> torch.Tensor:
        """Forward with Q-Q self-attention + cross-attention to hybrid memory.
        
        Args:
            q_x: [B, n, D] query stream input
            h_content: [B, n, D] content stream hidden state
            h_obs: [B, n, D] observation stream hidden state
            t_cond: [B, n, cond_dim] time conditioning
            rotary_cos_sin: RoPE embeddings (for seq_len=n)
            M_QQ_additive: [B, n, n] Q-Q self-attention mask (additive)
            M_hybrid_additive: [B, n, 2n] hybrid cross-attention mask (additive)
            
        Returns:
            Updated query hidden state [B, n, D]
        """
        bias_dropout_scale_fn = (
            bias_dropout_add_scale_fused_train if self.training
            else bias_dropout_add_scale_fused_inference
        )
        
        B, n, D = q_x.shape
        
        # Extract AdaLN parameters
        shift_self = scale_self = gate_self = None
        shift_cross = scale_cross = gate_cross = None
        shift_mlp = scale_mlp = gate_mlp = None
        shift_mlp2 = scale_mlp2 = gate_mlp2 = None
        
        if self.use_time and self.adaLN_modulation is not None and t_cond is not None:
            ada_params = self.adaLN_modulation(t_cond)
            (shift_self, scale_self, gate_self,
             shift_cross, scale_cross, gate_cross,
             shift_mlp, scale_mlp, gate_mlp,
             shift_mlp2, scale_mlp2, gate_mlp2) = ada_params.chunk(12, dim=2)
        
        cos, sin = rotary_cos_sin
        cos = cos.to(q_x.dtype)
        sin = sin.to(q_x.dtype)
        
        # =================================================================
        # Step 1: Q-Q Self-Attention
        # =================================================================
        x_before_self = q_x
        q_normed_self = self.q_self_norm(q_x)
        
        if shift_self is not None:
            q_normed_self = modulate_fused(q_normed_self, shift_self, scale_self)
        
        qkv_self = einops.rearrange(
            self.q_self_qkv(q_normed_self),
            'b s (three h d) -> b s three h d',
            three=3, h=self.n_heads
        )
        
        q_self = qkv_self[:, :, 0]
        k_self = qkv_self[:, :, 1]
        v_self = qkv_self[:, :, 2]
        
        q_self = apply_rotary_pos_emb_single(q_self, cos, sin)
        k_self = apply_rotary_pos_emb_single(k_self, cos, sin)
        
        x_self_attn = sdpa_attention_masked(q_self, k_self, v_self, M_QQ_additive, causal=False)
        
        if gate_self is not None:
            q_x = bias_dropout_scale_fn(
                self.q_self_out(x_self_attn), None, gate_self, x_before_self, self.dropout
            )
        else:
            scale = torch.ones(1, device=x_self_attn.device, dtype=x_self_attn.dtype)
            q_x = bias_dropout_scale_fn(
                self.q_self_out(x_self_attn), None, scale, x_before_self, self.dropout
            )
        
        # =================================================================
        # Step 2: Q -> [C||O] Cross-Attention
        # =================================================================
        x_before_cross = q_x
        q_normed_cross = self.q_cross_norm(q_x)
        
        if shift_cross is not None:
            q_normed_cross = modulate_fused(q_normed_cross, shift_cross, scale_cross)
        
        # Compute query for cross-attention
        q_cross = self.q_proj(q_normed_cross)
        q_cross = einops.rearrange(q_cross, 'b s (h d) -> b s h d', h=self.n_heads)
        q_cross = apply_rotary_pos_emb_single(q_cross, cos, sin)
        
        # Concatenate memory [C||O] -> [B, 2n, D]
        memory = torch.cat([h_content, h_obs], dim=1)
        memory_normed = self.mem_norm(memory)
        
        # Compute key-value from memory
        kv = self.kv_proj(memory_normed)
        kv = einops.rearrange(kv, 'b s (two h d) -> b s two h d', two=2, h=self.n_heads)
        k_cross = kv[:, :, 0]  # [B, 2n, h, d]
        v_cross = kv[:, :, 1]  # [B, 2n, h, d]
        
        # Apply RoPE to keys: use same positional indices for C_j and O_j
        k_C = k_cross[:, :n]  # [B, n, h, d]
        k_O = k_cross[:, n:]  # [B, n, h, d]
        k_C = apply_rotary_pos_emb_single(k_C, cos, sin)
        k_O = apply_rotary_pos_emb_single(k_O, cos, sin)
        k_cross = torch.cat([k_C, k_O], dim=1)  # [B, 2n, h, d]
        
        # Cross-attention with hybrid mask [B, n, 2n]
        x_cross_attn = sdpa_attention_masked(q_cross, k_cross, v_cross, M_hybrid_additive, causal=False)
        
        if gate_cross is not None:
            q_x = bias_dropout_scale_fn(
                self.cross_out(x_cross_attn), None, gate_cross, x_before_cross, self.dropout
            )
        else:
            scale = torch.ones(1, device=x_cross_attn.device, dtype=x_cross_attn.dtype)
            q_x = bias_dropout_scale_fn(
                self.cross_out(x_cross_attn), None, scale, x_before_cross, self.dropout
            )
        
        # =================================================================
        # Step 3: MLP
        # =================================================================
        x_before_mlp = q_x
        q_normed_mlp = self.norm_mlp(q_x)
        
        if shift_mlp is not None:
            q_normed_mlp = modulate_fused(q_normed_mlp, shift_mlp, scale_mlp)
        
        if gate_mlp is not None:
            q_x = bias_dropout_scale_fn(
                self.mlp(q_normed_mlp), None, gate_mlp, x_before_mlp, self.dropout
            )
        else:
            scale = torch.ones(1, device=q_x.device, dtype=q_x.dtype)
            q_x = bias_dropout_scale_fn(
                self.mlp(q_normed_mlp), None, scale, x_before_mlp, self.dropout
            )
        
        return q_x


class Campbell(nn.Module):
    """Three-stream transformer used by Campbell."""
    
    def __init__(self, config, vocab_size: int):
        super().__init__()
        if isinstance(config, dict):
            config = omegaconf.OmegaConf.create(config)
        
        self.config = config
        model_cfg = config.model if hasattr(config, "model") else config
        
        dim = model_cfg.hidden_size
        n_heads = model_cfg.n_heads
        self.vocab_size = vocab_size
        self.dim = dim
        self.n_heads = n_heads
        
        # Layer counts
        n_layers_mem = getattr(model_cfg, "n_layers_mem", 4)
        n_layers_q = getattr(model_cfg, "n_layers_q", 4)
        self.n_layers_mem = n_layers_mem
        self.n_layers_q = n_layers_q
        
        mlp_ratio = getattr(model_cfg, "mlp_ratio", 4)
        dropout = getattr(model_cfg, "dropout", 0.1)
        time_embed_dim = getattr(model_cfg, "time_freq_size", 256)
        cond_dim = getattr(model_cfg, "cond_dim", 128)
        self.cond_dim = cond_dim
        
        # Time conditioning options
        self.mem_use_time = getattr(model_cfg, "mem_use_time", False)
        self.query_use_time = getattr(model_cfg, "query_use_time", True)
        
        # Rank-time mode: replace tau with g(rank)
        self.rank_time_mode = getattr(model_cfg, "rank_time_mode", False)
        
        # Stream embeddings option
        self.use_stream_embeddings = getattr(model_cfg, "use_stream_embeddings", False)
        
        # Token IDs
        self.mask_token_id = getattr(model_cfg, "mask_token_id", vocab_size - 1)
        self.pad_token_id = getattr(model_cfg, "pad_token_id", 0)
        
        # Query initialization mode
        self.query_init_mode = getattr(model_cfg, "query_init_mode", "learned")
        
        # Debug mode for safety assertions
        self.debug_mode = getattr(model_cfg, "debug_mode", False)
        
        # =================================================================
        # Embeddings
        # =================================================================
        self.tok_embed = EmbeddingLayer(dim, vocab_size)
        
        # Stream embeddings (optional)
        if self.use_stream_embeddings:
            self.stream_embed_content = nn.Parameter(torch.zeros(1, 1, dim))
            self.stream_embed_obs = nn.Parameter(torch.zeros(1, 1, dim))
            self.stream_embed_query = nn.Parameter(torch.zeros(1, 1, dim))
            nn.init.normal_(self.stream_embed_content, std=0.02)
            nn.init.normal_(self.stream_embed_obs, std=0.02)
            nn.init.normal_(self.stream_embed_query, std=0.02)
        else:
            self.stream_embed_content = None
            self.stream_embed_obs = None
            self.stream_embed_query = None
        
        # Learned query init (query starts WITHOUT token content)
        if self.query_init_mode == "learned":
            self.query_init = nn.Parameter(torch.zeros(1, 1, dim))
            nn.init.normal_(self.query_init, std=0.02)
        else:
            self.query_init = None
        
        # RoPE
        self.rotary = Rotary(dim // n_heads)
        
        # Time conditioning
        self.sigma_map = TimestepEmbedder(cond_dim, frequency_embedding_size=time_embed_dim)
        
        # =================================================================
        # Memory Encoder (shared weights for C and O)
        # =================================================================
        self.memory_blocks = nn.ModuleList([
            MemoryBlock(
                dim, n_heads,
                cond_dim=cond_dim if self.mem_use_time else None,
                mlp_ratio=mlp_ratio,
                dropout=dropout,
                use_time=self.mem_use_time,
            )
            for _ in range(n_layers_mem)
        ])
        
        # =================================================================
        # Query Decoder (Q-Q self-attn + cross-attn to [C||O])
        # =================================================================
        self.query_blocks = nn.ModuleList([
            CampbellQueryBlock(
                dim, n_heads, cond_dim,
                mlp_ratio=mlp_ratio,
                dropout=dropout,
                use_time=self.query_use_time,
            )
            for _ in range(n_layers_q)
        ])
        
        # =================================================================
        # Output Head
        # =================================================================
        self.output_layer = DDiTFinalLayer(dim, vocab_size, cond_dim, adaLN=True)
    
    def _compute_rank_time(self, rank: torch.Tensor, n: int) -> torch.Tensor:
        """Map rank indices to deterministic times in ``[0, 1]``."""
        return (n - 1 - rank.float()) / max(n - 1, 1)
    
    def forward(
        self,
        x0_ids: torch.Tensor,
        xobs_ids: torch.Tensor,
        tau: torch.Tensor,
        rank: torch.Tensor | None = None,
        pad_mask: torch.Tensor | None = None,
        **kwargs,
    ) -> torch.Tensor:
        """Run the model on clean content, observed tokens, and event times."""
        B, n = x0_ids.shape
        device = x0_ids.device
        
        if rank is None:
            rank = compute_rank_from_tau(tau)

        M_le = build_M_le(rank, pad_mask)
        M_le_additive = normalize_attn_mask(M_le, mode="additive")

        M_ge = build_M_ge(rank, pad_mask)
        M_ge_additive = normalize_attn_mask(M_ge, mode="additive")

        M_QQ = build_M_QQ(rank, pad_mask)
        M_QQ_additive = normalize_attn_mask(M_QQ, mode="additive")

        M_hybrid = build_M_hybrid(rank, pad_mask)
        M_hybrid_additive = normalize_attn_mask(M_hybrid, mode="additive")

        if self.debug_mode:
            assert_hybrid_mask_safety(M_hybrid, rank)

        rotary_cos_sin = self.rotary(seq_len=n, device=device)

        if self.rank_time_mode:
            tau_used = self._compute_rank_time(rank, n)
        else:
            tau_used = tau

        tau_used = torch.clamp(tau_used, min=0.0, max=1.0)
        t_cond = F.silu(self.sigma_map(tau_used.reshape(-1))).view(B, n, -1)

        h_content = self.tok_embed(x0_ids)
        if self.stream_embed_content is not None:
            h_content = h_content + self.stream_embed_content
        if pad_mask is not None:
            h_content = h_content.masked_fill(pad_mask[..., None], 0.0)

        h_obs = self.tok_embed(xobs_ids)
        if self.stream_embed_obs is not None:
            h_obs = h_obs + self.stream_embed_obs
        if pad_mask is not None:
            h_obs = h_obs.masked_fill(pad_mask[..., None], 0.0)

        if self.query_init_mode == "learned" and self.query_init is not None:
            h_query = self.query_init.expand(B, n, -1).clone()
        else:
            h_query = torch.zeros(B, n, self.dim, device=device, dtype=h_content.dtype)
        if self.stream_embed_query is not None:
            h_query = h_query + self.stream_embed_query
        if pad_mask is not None:
            h_query = h_query.masked_fill(pad_mask[..., None], 0.0)

        for mem_block in self.memory_blocks:
            h_content = mem_block(
                h_content,
                t_cond if self.mem_use_time else None,
                rotary_cos_sin,
                M_le_additive,
            )
            h_obs = mem_block(
                h_obs,
                t_cond if self.mem_use_time else None,
                rotary_cos_sin,
                M_ge_additive,
            )

        for query_block in self.query_blocks:
            h_query = query_block(
                h_query,
                h_content=h_content,
                h_obs=h_obs,
                t_cond=t_cond if self.query_use_time else None,
                rotary_cos_sin=rotary_cos_sin,
                M_QQ_additive=M_QQ_additive,
                M_hybrid_additive=M_hybrid_additive,
            )

        return self.output_layer(h_query, t_cond)
    
    def forward_sampling(
        self,
        xcontent_ids: torch.Tensor,
        xobs_ids: torch.Tensor,
        tau: torch.Tensor,
        rank: torch.Tensor | None = None,
        pad_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Sampling-mode wrapper around ``forward``."""
        return self.forward(
            x0_ids=xcontent_ids,
            xobs_ids=xobs_ids,
            tau=tau,
            rank=rank,
            pad_mask=pad_mask,
        )
    
    def count_parameters(self) -> dict:
        """Count parameters by component."""
        stream_embed_count = sum(
            embed.numel()
            for embed in (
                self.stream_embed_content,
                self.stream_embed_obs,
                self.stream_embed_query,
            )
            if embed is not None
        )
        counts = {
            "tok_embed": sum(p.numel() for p in self.tok_embed.parameters()),
            "stream_embeds": stream_embed_count,
            "sigma_map": sum(p.numel() for p in self.sigma_map.parameters()),
            "memory_blocks": sum(p.numel() for p in self.memory_blocks.parameters()),
            "query_blocks": sum(p.numel() for p in self.query_blocks.parameters()),
            "output_layer": sum(p.numel() for p in self.output_layer.parameters()),
            "total": sum(p.numel() for p in self.parameters()),
        }
        if self.query_init is not None:
            counts["query_init"] = self.query_init.numel()
        return counts


# =============================================================================
# Sanity Tests
# =============================================================================

def sanity_check_campbell():
    """Run basic mask and forward-pass checks."""
    import torch
    
    print("=" * 60)
    print("Campbell Sanity Check")
    print("=" * 60)
    
    # Create minimal config
    config = {
        "model": {
            "hidden_size": 64,
            "n_heads": 4,
            "n_layers_mem": 2,
            "n_layers_q": 2,
            "mlp_ratio": 4,
            "dropout": 0.0,
            "cond_dim": 32,
            "time_freq_size": 64,
            "mem_use_time": False,
            "query_use_time": True,
            "rank_time_mode": False,
            "use_stream_embeddings": False,
            "query_init_mode": "learned",
            "debug_mode": True,  # Enable safety assertions
        }
    }
    
    vocab_size = 100
    B, n = 2, 16
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    print(f"Device: {device}")
    print(f"Batch size: {B}, Sequence length: {n}, Vocab size: {vocab_size}")
    
    # Create test inputs
    x0_ids = torch.randint(0, vocab_size - 1, (B, n), device=device)
    xobs_ids = torch.full((B, n), vocab_size - 1, device=device)  # All MASK
    tau = torch.rand(B, n, device=device)
    
    # =========================================================================
    # Test 1: Mask shapes and correctness
    # =========================================================================
    print(f"\n--- Test 1: Mask shapes ---")
    rank = compute_rank_from_tau(tau)
    
    M_le = build_M_le(rank, None)
    M_ge = build_M_ge(rank, None)
    M_QQ = build_M_QQ(rank, None)
    M_hybrid = build_M_hybrid(rank, None)
    
    print(f"M_le shape: {M_le.shape} (expected: [{B}, {n}, {n}])")
    print(f"M_ge shape: {M_ge.shape} (expected: [{B}, {n}, {n}])")
    print(f"M_QQ shape: {M_QQ.shape} (expected: [{B}, {n}, {n}])")
    print(f"M_hybrid shape: {M_hybrid.shape} (expected: [{B}, {n}, {2*n}])")
    
    assert M_le.shape == (B, n, n), "M_le shape mismatch!"
    assert M_ge.shape == (B, n, n), "M_ge shape mismatch!"
    assert M_QQ.shape == (B, n, n), "M_QQ shape mismatch!"
    assert M_hybrid.shape == (B, n, 2*n), "M_hybrid shape mismatch!"
    
    # =========================================================================
    # Test 2: M_QQ includes self (diagonal all True)
    # =========================================================================
    print(f"\n--- Test 2: M_QQ includes self (avoids empty keys) ---")
    M_QQ_diag = torch.diagonal(M_QQ, dim1=1, dim2=2)  # [B, n]
    assert M_QQ_diag.all(), "M_QQ diagonal should be all True (includes self)"
    print("  M_QQ diagonal is all True: PASS")
    
    # =========================================================================
    # Test 3: Hybrid mask C part is STRICT (diagonal all False)
    # =========================================================================
    print(f"\n--- Test 3: M_hybrid C part is STRICT (excludes self) ---")
    C_part = M_hybrid[:, :, :n]
    C_diag = torch.diagonal(C_part, dim1=1, dim2=2)
    assert not C_diag.any(), "M_hybrid C part diagonal should be all False"
    print("  M_hybrid C diagonal is all False: PASS")
    
    # =========================================================================
    # Test 4: Rank-0 has valid O keys but no C keys
    # =========================================================================
    print(f"\n--- Test 4: Rank-0 has valid keys in M_hybrid ---")
    rank_0_mask = (rank == 0)  # [B, n]
    O_part = M_hybrid[:, :, n:]
    
    for b in range(B):
        rank_0_positions = torch.where(rank_0_mask[b])[0]
        if len(rank_0_positions) > 0:
            pos = rank_0_positions[0].item()
            C_valid = C_part[b, pos].any().item()
            O_valid = O_part[b, pos].any().item()
            print(f"  Batch {b}, rank-0 at pos {pos}:")
            print(f"    M_hybrid C part has valid keys: {C_valid} (expected: False)")
            print(f"    M_hybrid O part has valid keys: {O_valid} (expected: True)")
            assert not C_valid, "Rank-0 should have no valid keys in C part!"
            assert O_valid, "Rank-0 should have valid keys in O part!"
    
    # =========================================================================
    # Test 5: Safety assertion function
    # =========================================================================
    print(f"\n--- Test 5: Safety assertion function ---")
    try:
        assert_hybrid_mask_safety(M_hybrid, rank)
        print("  assert_hybrid_mask_safety: PASS")
    except AssertionError as e:
        print(f"  assert_hybrid_mask_safety FAILED: {e}")
        raise
    
    # =========================================================================
    # Test 6: Campbell forward pass
    # =========================================================================
    print(f"\n--- Test 6: Campbell forward pass ---")
    model = Campbell(config, vocab_size).to(device)
    model.eval()
    
    params = model.count_parameters()
    print(f"Total parameters: {params['total']:,}")
    print(f"  Memory blocks: {params['memory_blocks']:,}")
    print(f"  Query blocks: {params['query_blocks']:,}")
    
    with torch.no_grad():
        logits = model(x0_ids, xobs_ids, tau)
    
    print(f"Logits shape: {logits.shape} (expected: [{B}, {n}, {vocab_size}])")
    assert logits.shape == (B, n, vocab_size), "Logits shape mismatch!"
    
    has_nan = torch.isnan(logits).any().item()
    has_inf = torch.isinf(logits).any().item()
    print(f"NaN in logits: {has_nan}")
    print(f"Inf in logits: {has_inf}")
    assert not has_nan, "NaN detected in logits!"
    
    # =========================================================================
    # Test 7: Campbell forward_sampling
    # =========================================================================
    print(f"\n--- Test 7: Campbell forward_sampling ---")
    xcontent_ids = torch.full((B, n), model.pad_token_id, device=device)
    with torch.no_grad():
        logits_sample = model.forward_sampling(xcontent_ids, xobs_ids, tau)
    
    print(f"Sampling logits shape: {logits_sample.shape}")
    has_nan = torch.isnan(logits_sample).any().item()
    print(f"NaN in sampling logits: {has_nan}")
    assert not has_nan, "NaN detected in sampling logits!"
    
    # =========================================================================
    # Test 8: Rank-time mode
    # =========================================================================
    print(f"\n--- Test 8: Rank-time mode ---")
    config_rank_time = {
        "model": {
            "hidden_size": 64,
            "n_heads": 4,
            "n_layers_mem": 2,
            "n_layers_q": 2,
            "mlp_ratio": 4,
            "dropout": 0.0,
            "cond_dim": 32,
            "time_freq_size": 64,
            "mem_use_time": False,
            "query_use_time": True,
            "rank_time_mode": True,  # Enable rank-time mode
            "use_stream_embeddings": False,
            "query_init_mode": "learned",
        }
    }
    model_rt = Campbell(config_rank_time, vocab_size).to(device)
    model_rt.eval()
    
    with torch.no_grad():
        logits_rt = model_rt(x0_ids, xobs_ids, tau)
    
    print(f"Rank-time mode logits shape: {logits_rt.shape}")
    has_nan = torch.isnan(logits_rt).any().item()
    print(f"NaN in rank-time logits: {has_nan}")
    assert not has_nan, "NaN detected in rank-time mode logits!"
    
    print("\n" + "=" * 60)
    print("All Campbell checks passed.")
    print("=" * 60)
    
    return True


# =============================================================================
# Exports
# =============================================================================

__all__ = [
    "Campbell",
    "MemoryBlock",
    "CampbellQueryBlock",
    "build_M_le",
    "build_M_ge",
    "build_M_QQ",
    "build_M_hybrid",
    "assert_hybrid_mask_safety",
    "sanity_check_campbell",
]


if __name__ == "__main__":
    sanity_check_campbell()
