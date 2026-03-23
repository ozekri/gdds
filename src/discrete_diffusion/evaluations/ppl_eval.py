"""Intrinsic validation perplexity evaluation for discrete diffusion checkpoints.

This script:
  - Loads a training config via Hydra (same as `python -m discrete_diffusion`)
  - Overrides dataset/model/algo via CLI (e.g. data=wikitext103 algo=mdlm model=small)
  - Loads a checkpoint specified by `eval.checkpoint_path`
  - Builds ONLY the validation dataloader for the chosen dataset
  - Runs a full pass over the validation set, computing:
        - avg NLL and PPL under the standard validation loss (train_mode=False)
        - avg NLL_true and PPL_true under the "true" loss (train_mode=True, respects loss_type)

Usage example:

  PYTHONPATH=src python -m discrete_diffusion.evaluations.ppl_eval \\
    data=wikitext103 \\
    data.cache_dir=$HOME/.cache/gdds/wikitext103 \\
    data.insert_valid_eos=False \\
    model=small \\
    algo=mdlm \\
    model.length=1024 \\
    loader.global_batch_size=128 \\
    loader.eval_global_batch_size=128 \\
    trainer.devices=1 \\
    eval.checkpoint_path=/path/to/checkpoint.ckpt
"""

from __future__ import annotations

import contextlib
import json
import math
import os
from pathlib import Path
from typing import Any, Dict

import hydra
import omegaconf
import torch
from omegaconf import DictConfig

from .. import utils
from ..data import get_dataloaders, get_tokenizer


# Match the main training entrypoint: configs directory is at the repo root
# (two levels above src/discrete_diffusion/, three above this file).
CONFIG_PATH = (Path(__file__).resolve().parents[3] / "configs").as_posix()


def _register_resolver(name, resolver):
    if omegaconf.OmegaConf.has_resolver(name):
        return
    omegaconf.OmegaConf.register_new_resolver(name, resolver)


def _mul_resolver(*args):
    import functools
    import operator

    return functools.reduce(operator.mul, args) if args else ValueError(
        "`mul` resolver requires at least one argument."
    )


# Register the same OmegaConf resolvers as the main CLI (__main__.py)
_register_resolver("cwd", lambda: Path().resolve().as_posix())
_register_resolver("device_count", torch.cuda.device_count)
_register_resolver("div_up", lambda x, y: (x + y - 1) // y)
_register_resolver("mul", _mul_resolver)
_register_resolver("sub", lambda x, y: x - y)


def _to_float(x: torch.Tensor | float) -> float:
    if isinstance(x, torch.Tensor):
        return float(x.detach().cpu().item())
    return float(x)


def _resolve_forward_process_config_from_checkpoint(config: DictConfig, ckpt: Dict[str, Any] | None):
    """Resolve forward-process config, preferring checkpoint values when present."""
    sik_path = Path(CONFIG_PATH) / "forward_process" / "sik_knn.yaml"
    if sik_path.is_file():
        base = omegaconf.OmegaConf.load(sik_path.as_posix())
    else:
        base = omegaconf.OmegaConf.create({})
    base_dict = omegaconf.OmegaConf.to_container(base, resolve=True) if base else {}

    fp_override_dict: Dict[str, Any] = {}
    if ckpt is not None:
        saved_fp = ckpt.get("hyper_parameters", {}).get("forward_process_config")
        if isinstance(saved_fp, dict) and saved_fp:
            fp_override_dict = dict(saved_fp)

    if not fp_override_dict:
        fp_override = omegaconf.OmegaConf.select(config, "algo.forward_process", default=None)
        if fp_override is not None:
            try:
                fp_override_dict = dict(
                    omegaconf.OmegaConf.to_container(fp_override, resolve=True) or {}
                )
            except Exception:
                fp_override_dict = {}

    merged = {**base_dict, **fp_override_dict}
    return omegaconf.OmegaConf.create(merged)


@hydra.main(version_base=None, config_path=CONFIG_PATH, config_name="config")
def main(config: DictConfig) -> Dict[str, Any]:
    """Compute intrinsic validation NLL / PPL of a checkpoint on a chosen dataset."""
    torch.set_float32_matmul_precision("high")
    torch.set_grad_enabled(False)

    if not os.environ.get("HF_HOME"):
        work = os.environ.get("WORK")
        if work:
            os.environ["HF_HOME"] = os.path.join(work, ".cache", "huggingface")
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    os.environ.setdefault("HF_DATASETS_OFFLINE", "1")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if not config.eval.checkpoint_path:
        raise ValueError("eval.checkpoint_path must be provided for ppl_eval.")
    # eval.checkpoint_path is typically already a string; just cast safely.
    ckpt_path = str(config.eval.checkpoint_path)
    if not utils.fsspec_exists(ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found at: {ckpt_path}")

    print("=" * 80)
    print("Intrinsic Validation Perplexity Evaluation (diffusion checkpoint)")
    print("=" * 80)
    print(f"Checkpoint: {ckpt_path}")
    print(f"Train dataset (config.data.train): {config.data.train}")
    print(f"Valid dataset (config.data.valid): {config.data.valid}")
    print(f"Model: {config.model}")
    print(f"Algo: {config.algo.name if hasattr(config.algo, 'name') else config.algo}")
    print("=" * 80)

    # ------------------------------------------------------------------
    # Load model weights from checkpoint WITHOUT unpickling old tokenizer
    # ------------------------------------------------------------------
    print("\nLoading model from checkpoint...")
    # Load raw checkpoint dict, but first patch GPT2 tokenizers for backwards compatibility
    #    with older checkpoints that expect `wrapped_decode` / `wrapped_batch_decode` attributes.
    try:
        from transformers import GPT2TokenizerFast, GPT2Tokenizer  # type: ignore
    except Exception:  # ImportError or other
        GPT2TokenizerFast = GPT2Tokenizer = None  # type: ignore

    def _ensure_wrapped_methods(tokenizer_cls):
        if tokenizer_cls is None:
            return
        if not hasattr(tokenizer_cls, "wrapped_decode"):
            def wrapped_decode(self, *args, **kwargs):
                # Fallback: delegate to current decode implementation
                return self.decode(*args, **kwargs)
            tokenizer_cls.wrapped_decode = wrapped_decode  # type: ignore[attr-defined]
        if not hasattr(tokenizer_cls, "wrapped_batch_decode"):
            def wrapped_batch_decode(self, *args, **kwargs):
                return self.batch_decode(*args, **kwargs)
            tokenizer_cls.wrapped_batch_decode = wrapped_batch_decode  # type: ignore[attr-defined]

    _ensure_wrapped_methods(GPT2TokenizerFast)
    _ensure_wrapped_methods(GPT2Tokenizer)

    # Now safely load the full checkpoint (we trust our own checkpoints).
    ckpt_loaded = torch.load(ckpt_path, map_location=device)
    ckpt_config = None
    if isinstance(ckpt_loaded, dict):
        ckpt_config = ckpt_loaded.get("hyper_parameters", {}).get("config", None)
    fp_cfg_for_model = None
    if ckpt_config is not None:
        if not omegaconf.OmegaConf.is_config(ckpt_config):
            ckpt_config = omegaconf.OmegaConf.create(ckpt_config)
        # For checkpoint evaluation, the checkpoint should define the effective
        # model/algo/noise/sampling setup. Keep runtime dataset/loader/eval
        # overrides from the CLI, but replace model-family-specific sections.
        for section in ("model", "algo", "noise", "sampling"):
            if ckpt_config.get(section, None) is not None:
                config[section] = omegaconf.OmegaConf.create(
                    omegaconf.OmegaConf.to_container(ckpt_config[section], resolve=False)
                )
        print("Using model/algo/noise/sampling config from checkpoint.")
        algo_target = str(omegaconf.OmegaConf.select(config, "algo._target_", default="") or "")
        if "gdds" in algo_target.lower():
            fp_cfg_for_model = _resolve_forward_process_config_from_checkpoint(config, ckpt_loaded)

    # Build tokenizer and algorithm class from the effective config
    tokenizer = get_tokenizer(config)
    algo_cls = hydra.utils.get_class(config.algo._target_)
    model = algo_cls(config, tokenizer=tokenizer)

    # Some checkpoints nest weights under "state_dict"; handle both.
    state_dict = ckpt_loaded.get("state_dict", ckpt_loaded) if isinstance(ckpt_loaded, dict) else ckpt_loaded
    strict_load = not (
        algo_cls.__name__ == "GDDSDiffusion"
        and isinstance(state_dict, dict)
        and "_sik_extra_embeddings" in state_dict
    )
    missing, unexpected = model.load_state_dict(state_dict, strict=strict_load)
    if missing or unexpected:
        print("Warning: state_dict load had mismatches:")
        if missing:
            print(f"  Missing keys ({len(missing)}): {missing[:10]}{' ...' if len(missing) > 10 else ''}")
        if unexpected:
            print(f"  Unexpected keys ({len(unexpected)}): {unexpected[:10]}{' ...' if len(unexpected) > 10 else ''}")
    if (
        algo_cls.__name__ == "GDDSDiffusion"
        and isinstance(state_dict, dict)
        and "_sik_extra_embeddings" in state_dict
    ):
        model.register_buffer("_sik_extra_embeddings", state_dict["_sik_extra_embeddings"].to(device))
    if fp_cfg_for_model is not None and hasattr(model, "_fp_cfg"):
        model._fp_cfg = fp_cfg_for_model
    model.to(device)
    model.eval()
    if algo_cls.__name__ == "GDDSDiffusion" and hasattr(model, "_setup_forward_process"):
        model._setup_forward_process()

    # Build ONLY validation dataloader for the chosen dataset
    print("\nBuilding validation dataloader...")
    _, valid_loader = get_dataloaders(
        config,
        tokenizer,
        skip_train=True,
        skip_valid=False,
    )

    if valid_loader is None:
        raise RuntimeError("Validation dataloader is None – check data.valid / loader config.")

    # Use same precision as training (bf16 by default) for comparable numerics
    precision = omegaconf.OmegaConf.select(config, "trainer.precision") or "bf16"
    use_bf16 = (
        str(precision).lower() in ("bf16", "bf16-mixed", "16-mixed")
        and device.type == "cuda"
    )
    if use_bf16:
        print("Using bf16 for validation (matches training precision).")
    else:
        print(f"Using full precision for validation (trainer.precision={precision}).")

    sum_nll = 0.0
    sum_tokens = 0.0
    sum_nll_true = 0.0
    sum_tokens_true = 0.0

    # For UDLM we want:
    #   - ppl      : use_path_loss = False, low_var = False
    #   - ppl_true : use_path_loss = False, low_var = True
    # which we implement by temporarily overriding model.loss_type and
    # model.udlm_loss_variant around the _loss() calls.
    algo_name = str(getattr(config.algo, "name", "")).lower()
    is_udlm = "udlm" in algo_name

    # UDLM's path-wise loss is memory-hungry, but we are explicitly disabling it
    # for evaluation. We still keep a small micro-batch size for extra safety.
    micro_bsz = 4 if is_udlm else None

    print("\nRunning validation pass...")
    autocast_ctx = (
        torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16)
        if use_bf16
        else contextlib.nullcontext()
    )
    with torch.no_grad():
        for batch_idx, batch in enumerate(valid_loader):
            x0_full = batch["input_ids"].to(device)
            valid_full = batch["attention_mask"].to(device)
            B = x0_full.size(0)

            def _accumulate_for_chunk(x0_chunk, valid_chunk):
                nonlocal sum_nll, sum_tokens, sum_nll_true, sum_tokens_true

                if is_udlm:
                    # Save original settings
                    old_loss_type = getattr(model, "loss_type", None)
                    old_udlm_variant = getattr(model, "udlm_loss_variant", None)

                    # 1) ppl: standard loss (no low_var, no path-wise)
                    model.loss_type = "elbo"  # anything != "low_var"
                    if hasattr(model, "udlm_loss_variant"):
                        model.udlm_loss_variant = "standard"
                    losses = model._loss(
                        x0_chunk,
                        valid_chunk,
                        train_mode=False,
                    )

                    # 2) ppl_true: low-var loss (no path-wise)
                    model.loss_type = "low_var"
                    if hasattr(model, "udlm_loss_variant"):
                        model.udlm_loss_variant = "standard"
                    losses_true = model._loss(
                        x0_chunk,
                        valid_chunk,
                        train_mode=True,
                    )

                    # Restore originals
                    if old_loss_type is not None:
                        model.loss_type = old_loss_type
                    if old_udlm_variant is not None and hasattr(model, "udlm_loss_variant"):
                        model.udlm_loss_variant = old_udlm_variant
                else:
                    # Default behaviour for non-UDLM algos:
                    # ppl uses train_mode=False; ppl_true uses train_mode=True.
                    losses = model._loss(
                        x0_chunk,
                        valid_chunk,
                        train_mode=False,
                    )
                    losses_true = model._loss(
                        x0_chunk,
                        valid_chunk,
                        train_mode=True,
                    )

                sum_nll += _to_float(losses.nlls)
                sum_tokens += _to_float(losses.num_tokens)
                sum_nll_true += _to_float(losses_true.nlls)
                sum_tokens_true += _to_float(losses_true.num_tokens)

            with autocast_ctx:
                if micro_bsz is None or B <= micro_bsz:
                    _accumulate_for_chunk(x0_full, valid_full)
                else:
                    for start in range(0, B, micro_bsz):
                        end = min(start + micro_bsz, B)
                        _accumulate_for_chunk(x0_full[start:end], valid_full[start:end])

            if (batch_idx + 1) % 50 == 0:
                print(f"  Processed {batch_idx + 1} batches...")

    if sum_tokens == 0 or sum_tokens_true == 0:
        raise RuntimeError("No tokens were counted during validation; check dataset and masks.")

    avg_nll = sum_nll / sum_tokens
    ppl = math.exp(avg_nll)
    avg_nll_true = sum_nll_true / sum_tokens_true
    ppl_true = math.exp(avg_nll_true)

    results: Dict[str, Any] = {
        "checkpoint_path": ckpt_path,
        "data_train": config.data.train,
        "data_valid": config.data.valid,
        "algo": omegaconf.OmegaConf.to_container(config.algo, resolve=True),
        "model": omegaconf.OmegaConf.to_container(config.model, resolve=True),
        "avg_nll": avg_nll,
        "ppl": ppl,
        "tokens": int(sum_tokens),
        "avg_nll_true": avg_nll_true,
        "ppl_true": ppl_true,
        "tokens_true": int(sum_tokens_true),
    }

    print("\n=== Validation metrics ===")
    print(f"  avg_nll       : {avg_nll:.6f}")
    print(f"  ppl           : {ppl:.4f}")
    print(f"  tokens        : {int(sum_tokens)}")
    print(f"  avg_nll_true  : {avg_nll_true:.6f}")
    print(f"  ppl_true      : {ppl_true:.4f}")
    print(f"  tokens_true   : {int(sum_tokens_true)}")
    print("==========================")

    # Optional: save JSON if user provided a path via eval.results_json_path
    if config.eval.get("results_json_path", None):
        out_path = Path(hydra.utils.to_absolute_path(config.eval.results_json_path))
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nSaved metrics to: {out_path}")

    return results


if __name__ == "__main__":
    main()
