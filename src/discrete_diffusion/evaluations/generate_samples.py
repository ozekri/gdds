"""Script to generate samples from a trained checkpoint.

This script loads a trained model checkpoint and generates samples using the
configured sampler. The output is saved as a PyTorch tensor (.pt) which can be
used for evaluation with the repository's downstream evaluation utilities.
"""

import hydra
import torch
import tqdm
from pathlib import Path
from omegaconf import OmegaConf
import transformers

from discrete_diffusion.data import get_tokenizer, wrap_tokenizer_decode_methods

# Workaround for checkpoints saved with wrapped tokenizers
# Patch tokenizer classes to handle missing 'wrapped_decode' attribute during unpickling
# This is backward compatible: old checkpoints (without wrapped_decode) work normally,
# and new checkpoints (with wrapped_decode) are handled gracefully.
_patched_tokenizer_classes = set()  # Track patched classes to avoid double-patching

def _patch_tokenizer_classes_for_unpickling():
    """Patch tokenizer classes to handle unpickling of checkpoints with wrapped tokenizers.
    
    This fix is backward compatible:
    - Old MDLM checkpoints: Don't reference wrapped_decode/wrapped_batch_decode, so __getattr__ 
      is never called for them. Normal attribute access works as before, no impact.
    - New SNAP checkpoints: If unpickling tries to access wrapped_decode or wrapped_batch_decode,
      __getattr__ handles them by returning self.decode/self.batch_decode, allowing unpickling to succeed.
    """
    def make_getattr_handler(cls):
        """Create a __getattr__ that handles wrapped_decode."""
        # Skip if already patched
        if cls in _patched_tokenizer_classes:
            return
        
        # Store original __getattr__ if it exists
        original_getattr = getattr(cls, '__getattr__', None)
        
        def patched_getattr(self, name):
            # Handle wrapped_decode and wrapped_batch_decode attributes that might be 
            # referenced in checkpoints saved with wrapped tokenizers (e.g., SNAP checkpoints)
            if name == 'wrapped_decode':
                # Return the decode method as a fallback
                return self.decode
            elif name == 'wrapped_batch_decode':
                # Return the batch_decode method as a fallback
                return self.batch_decode
            # Fall back to original __getattr__ if it exists
            if original_getattr is not None:
                return original_getattr(self, name)
            # Otherwise raise AttributeError (normal Python behavior)
            raise AttributeError(f"'{type(self).__name__}' object has no attribute '{name}'")
        
        # Only patch if __getattr__ doesn't exist or if we can safely wrap it
        if not hasattr(cls, '__getattr__'):
            cls.__getattr__ = patched_getattr
        elif original_getattr is not None:
            # Wrap the original __getattr__
            def wrapped_getattr(self, name):
                if name == 'wrapped_decode':
                    return self.decode
                elif name == 'wrapped_batch_decode':
                    return self.batch_decode
                return original_getattr(self, name)
            cls.__getattr__ = wrapped_getattr
        
        _patched_tokenizer_classes.add(cls)
    
    # Patch common tokenizer classes
    for cls in [transformers.GPT2TokenizerFast, transformers.GPT2Tokenizer]:
        make_getattr_handler(cls)

@hydra.main(config_path="../../../configs/eval", config_name="generate_samples", version_base="1.3")
def main(cfg):
    device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")
    torch.set_float32_matmul_precision('high')
    torch.set_grad_enabled(False)

    print(f"Loading checkpoint from {cfg.checkpoint_path}")
    checkpoint_path = hydra.utils.to_absolute_path(cfg.checkpoint_path)
    
    if not Path(checkpoint_path).exists():
        raise FileNotFoundError(f"Checkpoint not found at {checkpoint_path}")

    # Patch tokenizer classes before loading to handle wrapped tokenizers in checkpoints
    # This allows unpickling to succeed even if the checkpoint references 'wrapped_decode'
    _patch_tokenizer_classes_for_unpickling()
    
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    
    # Extract config from hyper_parameters
    if 'hyper_parameters' not in ckpt:
        raise ValueError("Checkpoint does not contain 'hyper_parameters'. Cannot load config.")
    
    if 'config' not in ckpt['hyper_parameters']:
         raise ValueError("Checkpoint hyper_parameters does not contain 'config'.")
         
    model_config = ckpt['hyper_parameters']['config']
    # Ensure it's an OmegaConf object
    if not isinstance(model_config, (dict, list, OmegaConf.get_type("DictConfig"), OmegaConf.get_type("ListConfig"))):
         model_config = OmegaConf.create(model_config)
    
    # Handle perm_batch_size override if provided
    if hasattr(cfg, 'perm_batch_size') and cfg.perm_batch_size is not None:
        if hasattr(model_config, 'sampling') and model_config.sampling is not None:
            model_config.sampling.perm_batch_size = cfg.perm_batch_size
            print(f"Overriding perm_batch_size to {cfg.perm_batch_size}")
        else:
            print(f"Warning: perm_batch_size={cfg.perm_batch_size} provided but no sampling config found")
    
    # Handle sampler config override if provided (for SNAP with annealed_block, etc.)
    if hasattr(cfg, 'sampler_config_path') and cfg.sampler_config_path is not None:
        sampler_config_path = hydra.utils.to_absolute_path(cfg.sampler_config_path)
        if Path(sampler_config_path).exists():
            print(f"Overriding sampler config with: {sampler_config_path}")
            sampler_override = OmegaConf.load(sampler_config_path)
            # Ensure sampling config exists
            if not hasattr(model_config, 'sampling') or model_config.sampling is None:
                model_config.sampling = OmegaConf.create({})
            # Convert existing sampling config to dict (in case it's a struct)
            # and merge with the override
            existing_sampling = OmegaConf.to_container(model_config.sampling, resolve=True)
            override_dict = OmegaConf.to_container(sampler_override, resolve=True)
            # Merge: override takes precedence
            merged_sampling = {**existing_sampling, **override_dict}
            # Set the merged config back (create new OmegaConf without struct mode)
            model_config.sampling = OmegaConf.create(merged_sampling)
            # Base algorithm resolves sampler from config.algo.sampler first, then config.sampling.sampler.
            # Keep them aligned so the override actually takes effect.
            if hasattr(model_config, 'algo') and model_config.algo is not None:
                model_config.algo.sampler = model_config.sampling.sampler
            print(f"Sampler override applied. New sampler: {model_config.sampling.sampler._target_}")
        else:
            raise FileNotFoundError(f"Sampler config not found at {sampler_config_path}")
    
    # Get tokenizer
    print("Loading tokenizer...")
    tokenizer = get_tokenizer(model_config)
    # Wrap tokenizer decode methods BEFORE loading checkpoint
    # This is critical: if the checkpoint was saved with a wrapped tokenizer,
    # PyTorch's unpickler needs the wrapped methods to be available
    wrap_tokenizer_decode_methods(tokenizer)
    
    # Identify algorithm class
    algo_target = model_config.algo._target_
    algo_cls = hydra.utils.get_class(algo_target)
    print(f"Detected algorithm class: {algo_cls.__name__}")
    
    # Load model following the same checkpoint/tokenizer compatibility path used
    # by the standalone sampling and evaluation utilities.
    # GDDS: model may not have _sik_extra_embeddings at init (it's created in _get_sik_embeddings);
    # we load with strict=False when present in checkpoint and restore the buffer after load.
    print("Loading model...")
    state = ckpt.get("state_dict") or {}
    strict_load = not (algo_cls.__name__ == "GDDSDiffusion" and "_sik_extra_embeddings" in state)
    model = algo_cls.load_from_checkpoint(
        checkpoint_path,
        config=model_config,
        tokenizer=tokenizer,
        map_location=device,
        strict=strict_load,
    )
    if "_sik_extra_embeddings" in state:
        model.register_buffer(
            "_sik_extra_embeddings",
            state["_sik_extra_embeddings"].to(device),
        )
        print("Restored _sik_extra_embeddings from checkpoint (PAD/MASK rows for SIK kernel).")

    model.to(device)
    model.eval()
    
    # Ensure model's tokenizer also has wrapped decode methods (in case it's a different instance)
    if hasattr(model, 'tokenizer') and model.tokenizer is not None:
        wrap_tokenizer_decode_methods(model.tokenizer)
    
    if cfg.torch_compile:
        print("Compiling model...")
        model = torch.compile(model)

    num_samples = cfg.num_samples
    batch_size = cfg.batch_size
    num_steps = cfg.num_steps
    
    print(f"Generating {num_samples} samples (batch_size={batch_size}, steps={num_steps or 'default'})")
    
    all_samples = []
    
    # Progress bar
    with tqdm.tqdm(total=num_samples, desc="Sampling", dynamic_ncols=True) as pbar:
        for i in range(0, num_samples, batch_size):
            current_batch_size = min(batch_size, num_samples - i)
            
            # Generate samples
            # We use model.generate_samples which delegates to the configured sampler
            samples = model.generate_samples(
                num_samples=current_batch_size,
                num_steps=num_steps
            )
            
            all_samples.append(samples.detach().cpu())
            pbar.update(current_batch_size)
            
            # Clear GPU cache between batches to prevent memory accumulation
            # This is especially important for memory-intensive samplers like annealed_block
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
    all_samples = torch.cat(all_samples, dim=0)
    
    # Save samples
    out_path = Path(hydra.utils.to_absolute_path(cfg.samples_path))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    
    torch.save(all_samples, out_path)
    print(f"Saved {len(all_samples)} samples to {out_path}")

    if cfg.get("save_text", False):
        print("Decoding samples to text...")
        texts = tokenizer.batch_decode(
            all_samples, 
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False  # Explicitly set to avoid FutureWarning
        )
        text_path = out_path.with_suffix('.txt')
        with open(text_path, 'w', encoding='utf-8') as f:
            for i, text in enumerate(texts):
                f.write(f"Sample {i}:\n{text}\n{'-'*80}\n")
        print(f"Saved text samples to {text_path}")

if __name__ == "__main__":
    main()
