"""Generalized Discrete Diffusion from Snapshots (GDDS)."""

from pathlib import Path
import hydra.utils
import omegaconf
import torch

from . import base as trainer_base
from ..forward_process import (
    SIKForwardProcess, 
    AbsorbingForwardProcess, 
    UniformForwardProcess,
    KNNKernel, 
    KeOpsKernel
)


class GDDSDiffusion(trainer_base.Diffusion):
    """Generalized Discrete Diffusion from Snapshots (GDDS)."""
    
    def __init__(self, config, tokenizer):
        # Needed early because TrainerBase.__init__ calls _resolve_sampler_config().
        self._fp_cfg = config.algo.forward_process
        super().__init__(config, tokenizer)
        
        self._validate_configuration()
        
        # Forward process setup: Source directly from the algo configuration
        self._forward_process = None
        self.mask_id = None
        
        # Cache for validation efficiency
        self._val_cache = None
        # Optional "carry over" trick: on non-jumped positions, carry x_t
        # deterministically to the output distribution.
        self.carry_over = bool(getattr(config.algo, 'carry_over', False))
        # Populated transiently inside nll() right before forward().
        self._active_jump_mask_for_output = None

    def _get_fp(self, key, default=None):
        return self._fp_cfg.get(key, default)

    def _use_carry_over(self):
        return self.carry_over and self.loss_type == 'elbo'

    def _validate_configuration(self):
        super()._validate_configuration()
        if self.parameterization != 'subs':
            raise ValueError('GDDSDiffusion expects algo.parameterization=subs')

        # Snapshot ELBO needs p(x0 | xt, t). Without time conditioning, the model
        # collapses to p(x0 | xt), which is a known bad regime for sampling quality.
        if self.loss_type == 'elbo' and not self.time_conditioning:
            raise ValueError(
                'GDDS with algo.loss_type=elbo requires algo.time_conditioning=True.'
            )

        # low_var does not require time conditioning (kept compatible with prior runs).

    def _resolve_sampler_config(self):
        """Pick a default sampler that matches the configured forward process.

        Priority:
        1) Respect explicit non-default `algo.sampler` override.
        2) Infer sampler from `algo.forward_process`.
        3) Fall back to base resolution.
        """
        algo_sampler = getattr(self.config.algo, 'sampler', None)
        algo_target = getattr(algo_sampler, '_target_', None)
        default_gdds_sampler = 'discrete_diffusion.sampling.gdds_sik_knn.GDDSSIKKNNSampler'

        # Preserve explicit user override while keeping the GDDS default stable.
        if algo_target and algo_target != default_gdds_sampler:
            return algo_sampler

        fp_cfg = getattr(self, '_fp_cfg', None)
        if fp_cfg is None:
            fp_cfg = getattr(self.config.algo, 'forward_process', {})
        fp_target = str(fp_cfg.get('_target_', ''))
        fp_name = str(fp_cfg.get('name', '')).lower()

        if ('AbsorbingForwardProcess' in fp_target) or (fp_name == 'absorbing'):
            return omegaconf.OmegaConf.create({
                '_target_': 'discrete_diffusion.sampling.absorbing.AbsorbingSampler'
            })

        if ('UniformForwardProcess' in fp_target) or (fp_name == 'uniform'):
            return omegaconf.OmegaConf.create({
                '_target_': 'discrete_diffusion.sampling.uniform.UniformSampler'
            })

        # SIK and variants keep the dedicated GDDS sampler.
        if any(k in fp_target for k in ['SIKForwardProcess', 'GDDSGauss', 'GDDSCosine']):
            return omegaconf.OmegaConf.create({'_target_': default_gdds_sampler})

        return super()._resolve_sampler_config()

    def _get_sik_embeddings(self, device):
        """Get token embeddings for SIK kernel."""
        source = self._get_fp('embedding_source', 'model')

        if source == 'model':
            if not hasattr(self.backbone, 'vocab_embed'):
                raise RuntimeError("Backbone needs 'vocab_embed' for embedding_source=model")
            emb = getattr(self.backbone.vocab_embed, 'embedding', getattr(self.backbone.vocab_embed, 'weight', None))
            if emb is None:
                emb = list(self.backbone.vocab_embed.parameters())[0]
            return emb.to(device).detach()

        if source == 'gpt2':
            import os
            raw_path = self._get_fp('gpt2_path')
            gpt2_load_path = os.path.expandvars(str(raw_path or self._get_fp('gpt2_model_name_or_path', 'gpt2'))).strip()
            from transformers import AutoModelForCausalLM
            model = AutoModelForCausalLM.from_pretrained(gpt2_load_path, local_files_only=True)
            model.resize_token_embeddings(self.vocab_size)
            embeddings = model.get_input_embeddings().weight.detach().to(device)
            
            extra_buf = getattr(self, '_sik_extra_embeddings', None)
            if extra_buf is not None and embeddings.size(0) > 50257:
                n_extra = min(extra_buf.size(0), embeddings.size(0) - 50257)
                embeddings[50257 : 50257 + n_extra] = extra_buf[:n_extra].to(device)
            return embeddings
        raise ValueError(f"Unknown embedding source: {source}")

    def _setup_forward_process(self):
        """Set up the forward process using the configured target."""
        if self._forward_process is not None:
            return

        device = next(self.backbone.parameters()).device
        target = self._fp_cfg.get("_target_", "")
        
        # Resolve if this is a SIK-based process (needs embedding extraction)
        is_sik = any(k in target for k in ["SIKForwardProcess", "GDDSGauss", "GDDSCosine"])
        
        if target and is_sik:
            embeddings = self._get_sik_embeddings(device)
            if embeddings.size(0) > 50257 and not hasattr(self, '_sik_extra_embeddings'):
                self.register_buffer('_sik_extra_embeddings', embeddings[50257:].clone())
            
            impl = self._get_fp("implementation", "knn")
            kernel_params = {
                "embeddings": embeddings,
                "epsilon": self._get_fp("epsilon", 0.01),
                "gamma": self._get_fp("gamma", 0.0),
                "metric": self._get_fp("metric", "gaussian"),
                "variable_bandwidth": self._get_fp("variable_bandwidth", True),
                "k_neighbors": self._get_fp("k_neighbors", 7),
            }
            
            if impl == "knn":
                kernel = KNNKernel(
                    **kernel_params,
                    top_k=self._get_fp("top_k", 64),
                    degree_chunk_size=self._get_fp("degree_chunk_size", 1000),
                )
            elif impl == "keops":
                kernel = KeOpsKernel(**kernel_params)
            else:
                raise ValueError(f"Unknown SIK implementation: {impl}")

            self._forward_process = SIKForwardProcess(
                tokenizer=self.tokenizer,
                schedule=self.noise,
                kernel=kernel,
                time_grid_size=self._get_fp("time_grid_size", 4096),
                time_eps=self._get_fp("time_eps", 1e-5),
                temperature_beta=self._get_fp("temperature_beta", 0.2),
                lambda_min=self._get_fp("lambda_min", 0.01),
                lambda_sigmoid_s=self._get_fp("lambda_sigmoid_s", 10.0),
                lambda_t0=self._get_fp("lambda_t0", 0.6),
                verbose=self._get_fp("verbose", False),
                name=self._get_fp("name", "sik"),
            ).to(device)
        else:
            # For Uniform or Absorbing, use standard Hydra instantiation
            self._forward_process = hydra.utils.instantiate(
                self._fp_cfg, tokenizer=self.tokenizer, schedule=self.noise, _recursive_=False
            ).to(device)
            
        # Dynamically sync attributes from the forward process
        if isinstance(self._forward_process, AbsorbingForwardProcess):
            self.mask_id = self._forward_process.mask_id
            
        # Ensure GDDS vocab_size matches exactly what the ForwardProcess uses
        self.vocab_size = self._forward_process.vocab_size
        
        # Register the correct limiting distribution provided by the process itself
        self.register_buffer(
            'limiting_distribution', 
            self._forward_process.get_limiting_distribution().to(device)
        )

    def nll(self, x0, current_accumulation_step=None, train_mode=False, valid_tokens=None):
        """Standard GDDS snapshot NLL with dual-metric logging."""
        if self._forward_process is None:
            self._setup_forward_process()
        
        trainer = getattr(self, "_trainer", None)
        is_validation = not (trainer is not None and trainer.training)
        
        # Use cache if available during validation
        if is_validation and self._val_cache is not None:
            if self._val_cache.get('x0_id') == x0.data_ptr():
                xt, jump_mask, sigma, log_x_theta = self._val_cache['xt'], self._val_cache['jump_mask'], self._val_cache['sigma'], self._val_cache['log_x_theta']
                return self.nll_per_token(log_x_theta, xt, x0, None, None, 
                                          low_var=train_mode and self.loss_type == 'low_var',
                                          valid_tokens=valid_tokens,
                                          train_mode=train_mode,
                                          jump_mask=jump_mask)
        
        t = self._sample_t(x0.shape[0], current_accumulation_step)
        if self.T > 0:
            t = (t * self.T).to(torch.int) / self.T + (1 / self.T)
        
        alpha_t = self.noise.alpha_t(t).unsqueeze(-1)
        sigma = self._sigma_from_alphat(alpha_t)
        
        # Standardized return_info API - GET THE EXACT JUMP MASK (N_t > 0)
        fp_res = self._forward_process(x0, t, return_info=True)
        xt, info = fp_res if isinstance(fp_res, tuple) else (fp_res, {})
        jump_mask = info.get('jump_mask', None)
        
        if getattr(self, 'ignore_bos', False):
            xt[:, 0] = x0[:, 0]
            if jump_mask is not None:
                jump_mask = jump_mask.clone()
                jump_mask[:, 0] = False
        
        if self._use_carry_over():
            self._active_jump_mask_for_output = jump_mask
        try:
            log_x_theta = self.forward(xt, sigma=sigma)
        finally:
            self._active_jump_mask_for_output = None

        if is_validation:
            self._val_cache = {'x0_id': x0.data_ptr(), 'xt': xt, 'jump_mask': jump_mask, 'sigma': sigma, 'log_x_theta': log_x_theta}
        
        return self.nll_per_token(log_x_theta, xt, x0, alpha_t, None, 
                                  low_var=train_mode and self.loss_type == 'low_var',
                                  valid_tokens=valid_tokens,
                                  train_mode=train_mode,
                                  jump_mask=jump_mask)

    def nll_per_token(self, log_x_theta, xt, x0, alpha_t, dalpha_t, low_var=False, valid_tokens=None, train_mode=False, jump_mask=None):
        """Compute the NLL per token for GDDS."""
        log_p_x0 = torch.gather(log_x_theta, -1, x0[:, :, None]).squeeze(-1)
        
        # Unweighted loss masked for padding
        loss_unweighted = -log_p_x0 * (valid_tokens.to(log_p_x0.dtype) if valid_tokens is not None else 1.0)
        
        # Metric logging pass
        if not train_mode:
            # Fallback for metric logging only if process info is unavailable.
            jm_metrics = jump_mask if jump_mask is not None else (xt != x0)
            
            num_tokens = valid_tokens.sum() if valid_tokens is not None else log_p_x0.numel()
            self.metrics.update_valid(loss_unweighted.sum(), num_tokens)
            
            # Denoising PPL: Use the exact jump mask from the process if available
            jump_mask_valid = jm_metrics & (valid_tokens.bool() if valid_tokens is not None else True)
            loss_denoising = loss_unweighted * jump_mask_valid.to(loss_unweighted.dtype)
            num_jumped = jump_mask_valid.sum()
            if num_jumped > 0:
                self.metrics.update_valid_denoising(loss_denoising.sum(), num_jumped)
        
        if low_var:
            if jump_mask is None:
                raise ValueError("GDDS low_var requires an explicit jump_mask from the forward process (N_t > 0).")
            # OPTIMIZATION: Only backprop through tokens that actually jumped (N_t > 0)
            return -log_p_x0 * jump_mask.to(log_p_x0.dtype)
        
        # True Snapshot NLL: computes loss on all tokens
        return -log_p_x0

    def on_fit_start(self):
        super().on_fit_start()
        if self._forward_process is None:
            self._setup_forward_process()
        sampler = self._create_sampler()
        if getattr(self, "global_rank", 0) == 0:
            fp_name = type(self._forward_process).__name__
            sampler_name = type(sampler).__name__ if sampler is not None else "None"
            print(
                f"[GDDS] setup: forward_process={fp_name}, sampler={sampler_name}, "
                f"loss_type={self.loss_type}"
            )

    def prior_sample(self, *batch_dims):
        if self._forward_process is None:
            self._setup_forward_process()
        return self._forward_process.sample_prior(*batch_dims).to(self.device)

    def q_xt(self, x, t):
        if self._forward_process is None:
            self._setup_forward_process()
        out = self._forward_process(x, t)
        xt = out[0] if isinstance(out, (tuple, list)) else out
        if getattr(self, 'ignore_bos', False):
            xt[:, 0] = x[:, 0]
        return xt

    def _process_model_output(self, model_output, xt, sigma):
        del sigma
        log_probs = torch.log_softmax(model_output, dim=-1)

        if not self._use_carry_over():
            return log_probs

        jump_mask = self._active_jump_mask_for_output
        if jump_mask is None:
            # e.g. sampling/inference: no ground-truth jump information available
            return log_probs

        if jump_mask.shape != xt.shape:
            raise ValueError(
                f"jump_mask shape mismatch: {jump_mask.shape} vs xt {xt.shape}"
            )

        # MDLM-style deterministic copy branch on non-jumped positions:
        # log p(xt)=0 and log p(other)=-inf. This zeroes gradients there.
        copy_logits = torch.full_like(log_probs, self.neg_infinity)
        copy_logits = torch.scatter(copy_logits, -1, xt[..., None], 0.0)
        non_jump = (~jump_mask).unsqueeze(-1)
        return torch.where(non_jump, copy_logits, log_probs)
