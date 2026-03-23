"""Generative perplexity evaluation against a reference language model.

This script reports cross-perplexity under a reference LM such as GPT-2.
It supports token-ID scoring when tokenizers are compatible, text-based
scoring via decode-and-retokenize, and summary statistics for generated
samples and reference text.
"""

from __future__ import annotations

import itertools
import json
import os
import random
import re
import socket
import subprocess
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import hydra
import numpy as np
import torch
import torch.nn.functional as F
from omegaconf import DictConfig, OmegaConf
from transformers import AutoModelForCausalLM, AutoTokenizer

from .eval_metrics import (
    WorstTokenInfo,
    check_tokenizer_compatibility,
    compute_artifact_metrics,
    compute_bootstrap_ci,
    compute_byte_level_validity,
    compute_embedding_distance,
    compute_enhanced_diversity_metrics,
    compute_entropy_metrics,
    compute_longest_repeated_substring,
    compute_mauve_score,
    compute_nll_tail_stats,
    compute_robust_metrics,
    compute_tail_contribution,
    compute_token_type_stratified_nll,
    extract_worst_tokens,
)


def _default_cache_root() -> str:
    """Resolve the shared dataset cache root.

    Precedence:
      1) GDDS_DATA_DIR environment variable
      2) configs/paths/local.yaml
      3) configs/paths/default.yaml
    """
    root = os.environ.get("GDDS_DATA_DIR")
    if root:
        return root

    config_root = Path(__file__).resolve().parents[3] / "configs" / "paths"
    local_path = config_root / "local.yaml"
    default_path = config_root / "default.yaml"

    for path in (local_path, default_path):
        if path.is_file():
            cfg = OmegaConf.load(path)
            resolved = OmegaConf.to_container(cfg, resolve=True)
            cache_root = resolved.get("cache_root") if isinstance(resolved, dict) else None
            if cache_root:
                return str(cache_root)

    return os.path.join(os.path.expanduser("~"), ".cache", "gdds")


def _default_dataset_cache_dir(dataset_name: str) -> str:
    """Portable default cache location for cached evaluation datasets."""
    root = _default_cache_root()
    return os.path.join(root, "openwebtext" if "openwebtext" in dataset_name else dataset_name)


# =============================================================================
# Token-ID Degeneracy / Mode-Collapse Diagnostics
# =============================================================================


def _token_ngrams(seq: np.ndarray, n: int) -> List[Tuple[int, ...]]:
    """Extract token n-grams from a 1D sequence of token IDs.
    
    Args:
        seq: 1D numpy array of token IDs (shape [T])
        n: n-gram order (e.g., 4 for 4-grams)
    
    Returns:
        List of n-gram tuples. Returns empty list if sequence is shorter than n.
    """
    if len(seq) < n:
        return []
    return [tuple(seq[i : i + n].tolist()) for i in range(len(seq) - n + 1)]


def compute_token_id_degeneracy_metrics(
    token_ids: np.ndarray,
    ngram_n: int = 4,
    pair_sample: int = 2000,
    seed: int = 0,
) -> Dict[str, Any]:
    """Compute degeneracy/mode-collapse metrics directly in token-ID space.
    
    This avoids decode/retokenize round-trip issues by operating entirely on
    integer token IDs.
    
    Args:
        token_ids: 2D numpy array of shape [N, T] containing token IDs
        ngram_n: n-gram order for repetition and Jaccard metrics (default: 4)
        pair_sample: Number of random distinct pairs to sample for inter-sample
                     Jaccard similarity (default: 2000). If fewer pairs exist,
                     uses all pairs.
        seed: Random seed for pair sampling (default: 0)
    
    Returns:
        Dictionary with keys:
        - token_id_unique_ratio: fraction of unique sequences (exact match)
        - token_id_duplicate_rate: 1 - unique_ratio
        - rep_ngram_frac_mean, _median, _p95, _max: aggregated repeated n-gram
          fraction across samples (intra-sample repetition)
        - rep_ngram_n: the n-gram order used
        - inter_sample_jaccard_mean: mean Jaccard similarity between n-gram sets
          of sampled pairs
        - pair_sample: actual number of pairs used
    """
    N, T = token_ids.shape
    
    # -------------------------------------------------------------------------
    # 1. Sequence-level uniqueness (exact match in token IDs)
    # -------------------------------------------------------------------------
    # Convert each row to a hashable tuple for deduplication
    unique_seqs = set()
    for i in range(N):
        seq_tuple = tuple(token_ids[i].tolist())
        unique_seqs.add(seq_tuple)
    
    num_unique = len(unique_seqs)
    unique_ratio = num_unique / N if N > 0 else 1.0
    duplicate_rate = 1.0 - unique_ratio
    
    # -------------------------------------------------------------------------
    # 2. Intra-sample repetition: repeated n-gram fraction per sample
    # -------------------------------------------------------------------------
    # For each sample: rep_frac = 1 - (# unique n-grams / # total n-grams)
    rep_fracs = []
    ngram_sets_per_sample: List[set] = []  # Precompute for Jaccard
    
    for i in range(N):
        ngrams = _token_ngrams(token_ids[i], ngram_n)
        total_ngrams = len(ngrams)
        
        if total_ngrams == 0:
            # Sequence too short for any n-grams
            rep_fracs.append(0.0)
            ngram_sets_per_sample.append(set())
        else:
            unique_ngrams = set(ngrams)
            num_unique_ngrams = len(unique_ngrams)
            rep_frac = 1.0 - (num_unique_ngrams / total_ngrams)
            rep_fracs.append(rep_frac)
            ngram_sets_per_sample.append(unique_ngrams)
    
    rep_fracs_arr = np.array(rep_fracs, dtype=np.float64)
    
    if len(rep_fracs_arr) > 0:
        rep_ngram_frac_mean = float(np.mean(rep_fracs_arr))
        rep_ngram_frac_median = float(np.median(rep_fracs_arr))
        rep_ngram_frac_p95 = float(np.percentile(rep_fracs_arr, 95))
        rep_ngram_frac_max = float(np.max(rep_fracs_arr))
    else:
        rep_ngram_frac_mean = 0.0
        rep_ngram_frac_median = 0.0
        rep_ngram_frac_p95 = 0.0
        rep_ngram_frac_max = 0.0
    
    # -------------------------------------------------------------------------
    # 3. Inter-sample similarity: Jaccard on n-gram sets of sampled pairs
    # -------------------------------------------------------------------------
    # J(A, B) = |A ∩ B| / |A ∪ B|
    # Sample `pair_sample` random distinct pairs (or all pairs if fewer exist)
    
    # Total possible pairs: N * (N - 1) / 2
    max_pairs = N * (N - 1) // 2
    actual_pair_sample = min(pair_sample, max_pairs)
    
    jaccard_sims = []
    
    if actual_pair_sample > 0 and N >= 2:
        rng = random.Random(seed)
        
        if actual_pair_sample >= max_pairs:
            # Use all pairs
            pairs = list(itertools.combinations(range(N), 2))
        else:
            # Sample pairs without replacement
            # Use reservoir-like sampling for efficiency
            all_pairs_iter = itertools.combinations(range(N), 2)
            pairs = []
            for idx, pair in enumerate(all_pairs_iter):
                if idx < actual_pair_sample:
                    pairs.append(pair)
                else:
                    # Reservoir sampling
                    j = rng.randint(0, idx)
                    if j < actual_pair_sample:
                        pairs[j] = pair
        
        for (i, j) in pairs:
            set_i = ngram_sets_per_sample[i]
            set_j = ngram_sets_per_sample[j]
            
            # Handle empty sets
            if len(set_i) == 0 and len(set_j) == 0:
                # Both empty: define Jaccard as 1.0 (identical empty sets)
                jaccard = 1.0
            elif len(set_i) == 0 or len(set_j) == 0:
                # One empty, one not: Jaccard = 0.0
                jaccard = 0.0
            else:
                intersection = len(set_i & set_j)
                union = len(set_i | set_j)
                jaccard = intersection / union if union > 0 else 0.0
            
            jaccard_sims.append(jaccard)
        
        inter_sample_jaccard_mean = float(np.mean(jaccard_sims)) if jaccard_sims else 0.0
    else:
        inter_sample_jaccard_mean = 0.0
        actual_pair_sample = 0
    
    return {
        "token_id_unique_ratio": unique_ratio,
        "token_id_duplicate_rate": duplicate_rate,
        "rep_ngram_frac_mean": rep_ngram_frac_mean,
        "rep_ngram_frac_median": rep_ngram_frac_median,
        "rep_ngram_frac_p95": rep_ngram_frac_p95,
        "rep_ngram_frac_max": rep_ngram_frac_max,
        "rep_ngram_n": ngram_n,
        "inter_sample_jaccard_mean": inter_sample_jaccard_mean,
        "pair_sample": actual_pair_sample,
    }


def _check_token_artifact_flags(token_str: str) -> Dict[str, bool]:
    """Check for various artifact patterns in a token string."""
    flags = {
        "has_unicode_replacement": ("\ufffd" in token_str or "\uFFFD" in token_str),
        "non_ascii": any(ord(c) > 127 for c in token_str),
        "weird_whitespace": bool(re.search(r"[\t\r\n]|  +", token_str)),
        "long_alnum_run": bool(re.search(r"[A-Za-z0-9_./-]{20,}", token_str)),
        "broken_markup_hint": bool(re.search(r"\[|\]|UPDATE|Step|paywall|http|www", token_str, re.IGNORECASE)),
    }
    return flags


def _compute_quarter_stats(
    nlls: List[float],
    token_ids: List[int],
    tokenizer: AutoTokenizer,
) -> Dict[str, Any]:
    """Compute statistics for each quarter of the sequence."""
    n = len(nlls)
    if n == 0:
        return {}
    
    quarter_size = n // 4
    quarters = []
    
    for q in range(4):
        start_idx = q * quarter_size
        end_idx = (q + 1) * quarter_size if q < 3 else n
        quarter_nlls = nlls[start_idx:end_idx]
        quarter_token_ids = token_ids[start_idx:end_idx] if token_ids else []
        
        if len(quarter_nlls) == 0:
            quarters.append({
                "mean_nll": float("nan"),
                "median_nll": float("nan"),
                "ppl": float("nan"),
                "max_nll": float("nan"),
                "max_nll_token_str": "",
            })
            continue
        
        mean_nll = float(np.mean(quarter_nlls))
        median_nll = float(np.median(quarter_nlls))
        ppl = float(np.exp(mean_nll))
        max_nll = float(np.max(quarter_nlls))
        
        # Find token string with max NLL
        max_idx = np.argmax(quarter_nlls)
        max_token_id = quarter_token_ids[max_idx] if max_idx < len(quarter_token_ids) else 0
        max_token_str = tokenizer.decode([max_token_id], skip_special_tokens=False, clean_up_tokenization_spaces=False)
        
        quarters.append({
            "mean_nll": mean_nll,
            "median_nll": median_nll,
            "ppl": ppl,
            "max_nll": max_nll,
            "max_nll_token_str": repr(max_token_str),
        })
    
    return {"quarters": quarters}


def _extract_worst_tokens_with_flags(
    nll_tensor: torch.Tensor,
    input_ids: torch.Tensor,
    valid_mask: torch.Tensor,
    tokenizer: AutoTokenizer,
    top_k: int = 50,
    context_tokens: int = 10,
    batch_offset: int = 0,
) -> List[Dict[str, Any]]:
    """Extract worst tokens with artifact flags and token context window."""
    # Flatten and get indices
    valid_nlls = nll_tensor[valid_mask.bool()]
    if len(valid_nlls) == 0:
        return []
    
    valid_indices = torch.nonzero(valid_mask.bool(), as_tuple=False)
    
    # Get top-K worst tokens
    k = min(top_k, len(valid_nlls))
    top_k_indices = torch.topk(valid_nlls, k=k).indices
    
    worst_tokens = []
    for rank, idx in enumerate(top_k_indices, 1):
        batch_idx_local, token_pos = valid_indices[idx.item()].cpu().tolist()
        batch_idx_global = batch_offset + batch_idx_local
        # token_pos is in labels (shifted), so the actual token is at token_pos + 1 in input_ids
        token_id = input_ids[batch_idx_local, token_pos + 1].item()
        nll_val = valid_nlls[idx].item()
        
        # Get token string
        token_str = tokenizer.decode([token_id], skip_special_tokens=False, clean_up_tokenization_spaces=False)
        
        # Get context tokens (previous 10 + current + next 10)
        seq_len = input_ids.shape[1]
        context_start = max(0, token_pos + 1 - context_tokens)
        context_end = min(seq_len, token_pos + 2 + context_tokens)
        context_token_ids = input_ids[batch_idx_local, context_start:context_end].tolist()
        context_decoded = tokenizer.decode(context_token_ids, skip_special_tokens=False, clean_up_tokenization_spaces=False)
        
        # Find position of current token in context
        tokens_before = input_ids[batch_idx_local, context_start:token_pos + 1].tolist()
        decoded_before = tokenizer.decode(tokens_before, skip_special_tokens=False, clean_up_tokenization_spaces=False)
        token_start_in_context = len(decoded_before)
        tokens_up_to = input_ids[batch_idx_local, context_start:token_pos + 2].tolist()
        decoded_up_to = tokenizer.decode(tokens_up_to, skip_special_tokens=False, clean_up_tokenization_spaces=False)
        token_end_in_context = len(decoded_up_to)
        
        # Split context into before and after
        context_before_str = context_decoded[:token_start_in_context]
        context_after_str = context_decoded[token_end_in_context:]
        
        # Check artifact flags
        artifact_flags = _check_token_artifact_flags(token_str)
        
        worst_tokens.append({
            "rank": rank,
            "sample_idx": batch_idx_global,
            "token_idx": token_pos,
            "token_id": int(token_id),
            "token_str": repr(token_str),
            "nll": float(nll_val),
            "artifact_flags": artifact_flags,
            "context_before": repr(context_before_str),
            "context_after": repr(context_after_str),
            "full_context": repr(context_decoded),
        })
    
    return worst_tokens


def _check_artifact_sensitivity(
    texts: List[str],
    tokenizer: AutoTokenizer,
    model: AutoModelForCausalLM,
    device: torch.device,
    batch_size: int,
    max_length: int,
    first_chunk_only: bool,
    ignore_prefix_tokens: int,
    prompt_len: Optional[int],
) -> Dict[str, Any]:
    """Check if PPL changes significantly after sanitizing artifacts."""
    # Sanitize texts
    sanitized_texts = []
    for text in texts:
        # Normalize whitespace (replace all whitespace sequences with single space)
        text = re.sub(r"\s+", " ", text)
        # Remove/replace Unicode replacement characters
        text = text.replace("\ufffd", "").replace("\uFFFD", "")
        # Strip non-ASCII (optional - might be too aggressive)
        # For now, just remove replacement chars
        sanitized_texts.append(text)
    
    # Rescore sanitized texts
    try:
        sanitized_metrics, _, _ = _score_with_reference_model(
            texts=sanitized_texts,
            model=model,
            tokenizer=tokenizer,
            device=device,
            batch_size=batch_size,
            max_length=max_length,
            retokenize=True,
            first_chunk_only=first_chunk_only,
            ignore_prefix_tokens=ignore_prefix_tokens,
            prompt_len=prompt_len,
        )
        
        return {
            "sanitized_ppl": sanitized_metrics.get("ppl", float("nan")),
            "sanitized_avg_nll": sanitized_metrics.get("avg_nll", float("nan")),
        }
    except Exception as e:
        return {
            "error": str(e),
            "sanitized_ppl": None,
            "sanitized_avg_nll": None,
        }
from ..data.loaders import wrap_tokenizer_decode_methods


def _get_git_hash() -> Optional[str]:
    """Get current git commit hash if available."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=False,
            timeout=5,
        )
        if result.returncode == 0:
            return result.stdout.strip()
    except Exception:
        pass
    return None


def _load_samples(samples_path: str) -> np.ndarray:
    """Load samples from .pt, .npz, or .json file."""
    path = Path(hydra.utils.to_absolute_path(samples_path))
    if not path.exists():
        raise FileNotFoundError(f"Samples not found at {path}")

    if path.suffix == ".pt":
        z_ts = torch.load(path, weights_only=True)
        if isinstance(z_ts, torch.Tensor):
            arr = z_ts.detach().cpu()
        elif isinstance(z_ts, dict):
            if "input_ids" in z_ts:
                arr = z_ts["input_ids"].detach().cpu() if isinstance(z_ts["input_ids"], torch.Tensor) else torch.tensor(z_ts["input_ids"])
            elif "samples" in z_ts:
                arr = z_ts["samples"].detach().cpu() if isinstance(z_ts["samples"], torch.Tensor) else torch.tensor(z_ts["samples"])
            else:
                raise ValueError(f"Unsupported .pt dict structure. Keys: {list(z_ts.keys())}")
        else:
            raise ValueError(f"Unsupported .pt structure; expected Tensor or dict, got {type(z_ts)}")
        if arr.ndim == 3 and arr.shape[1] == 1:
            arr = arr.squeeze(1)
        return arr.numpy()

    if path.suffix == ".npz":
        content = np.load(path)
        if "samples" not in content:
            raise KeyError(".npz must contain 'samples' key")
        return content["samples"]

    if path.suffix == ".json":
        from ..utils import utils as _utils
        with open(path, "r") as f:
            payload = json.load(f)
        if "np_tokens_b64" not in payload:
            raise KeyError(".json must contain 'np_tokens_b64' key")
        arr = _utils.base64_to_np(payload["np_tokens_b64"])
        return arr

    raise ValueError(f"Unsupported samples format: {path.suffix}")


def _load_real_text_samples(
    dataset_name: str,
    num_samples: int,
    tokenizer: AutoTokenizer,
    cache_dir: Optional[str] = None,
) -> List[str]:
    """Load real text samples from a dataset for baseline comparison.
    
    CRITICAL: This function must return RAW TEXT STRINGS, not decoded token IDs.
    If the cached dataset contains token IDs from a different tokenizer (e.g., training tokenizer),
    decoding them with the reference tokenizer produces nonsense.
    
    Prefer "text" field if available. Only decode "input_ids" if we're certain the tokenizer matches.
    """
    try:
        from ..data.loaders import get_dataset
        from omegaconf import OmegaConf
    except ImportError:
        raise ImportError("Cannot import get_dataset. Make sure data loaders are available.")
    
    # Determine cache directory using the repo's portable default.
    if cache_dir is None:
        cache_dir = _default_dataset_cache_dir(dataset_name)
    
    # Determine mode (same logic as get_dataloaders in loaders.py)
    # For validation datasets, use "validation" mode (not "valid")
    # This matches how training scripts cache the dataset
    if "valid" in dataset_name:
        mode = "validation"  # Matches cached filename: openwebtext-valid_validation_bs1024_wrapped.dat
    elif "test" in dataset_name:
        mode = "test"
    else:
        mode = "train"
    
    # Load dataset from the configured cache location.
    print(f"Loading dataset '{dataset_name}' from cache: {cache_dir}")
    print(f"  Mode: {mode}")
    print("  Note: using the configured GDDS dataset cache directory")
    
    try:
        # Try to load directly from cache first (same logic as get_dataset)
        # Import utils from the correct location (same as loaders.py uses: from .. import utils)
        from .. import utils as data_utils
        try:
            import datasets
        except ImportError:
            raise ImportError("datasets library required. Install with: pip install datasets")
        
        # Build the expected cached filename (same logic as get_dataset)
        eos_tag = ""  # insert_eos=True, so no tag
        special_tag = ""  # insert_special_tokens=True, so no tag  
        filename = f"{dataset_name}_{mode}_bs1024_wrapped{eos_tag}{special_tag}.dat"
        cached_path = os.path.join(cache_dir, filename)
        
        if data_utils.fsspec_exists(cached_path):
            print(f"  Found cached dataset file: {cached_path}")
            print(f"  Loading directly from cache (no Hugging Face Hub access needed)")
            # Load directly from disk to avoid any Hugging Face Hub calls
            dataset = datasets.load_from_disk(cached_path).with_format("torch")
            dataset_len = len(dataset) if hasattr(dataset, "__len__") else "unknown"
            print(f"  ✓ Successfully loaded dataset from cache (length: {dataset_len})")
        else:
            # Cached file doesn't exist, try get_dataset (which will try to generate it)
            print(f"  Cached file not found at: {cached_path}")
            print(f"  Attempting to load via get_dataset (will use cache if available)...")
            dataset = get_dataset(
                dataset_name=dataset_name,
                tokenizer=tokenizer,
                wrap=True,
                mode=mode,
                cache_dir=cache_dir,
                insert_eos=True,
                insert_special_tokens=True,
                block_size=1024,
                streaming=False,
                num_proc=1,  # Use single process to avoid issues
            )
            dataset_len = len(dataset) if hasattr(dataset, "__len__") else "unknown"
            print(f"  ✓ Successfully loaded dataset (length: {dataset_len})")
    except Exception as e:
        error_msg = str(e)
        print(f"  ✗ Error loading dataset: {error_msg}")
        
        # Provide helpful error message
        if "Offline mode" in error_msg or "Hugging Face Hub" in error_msg:
            # The dataset should be pre-cached. Check what file it's looking for
            expected_filename = f"{dataset_name}_{mode}_bs1024_wrapped.dat"
            expected_path = os.path.join(cache_dir, expected_filename)
            
            raise RuntimeError(
                f"Failed to load dataset '{dataset_name}' from cache directory '{cache_dir}'. "
                f"The validation dataset should be available in this cache location. "
                f"\n"
                f"Default cache root: ${'{'}HOME{'}'}/.cache/gdds or $GDDS_DATA_DIR\n"
                f"Expected cache location: {cache_dir}\n"
                f"Expected cached file: {expected_path}\n"
                f"\n"
                f"If the file doesn't exist, you may need to cache the dataset first with the same tokenizer/block-size settings.\n"
                f"\n"
                f"Original error: {error_msg}"
            )
        else:
            raise
    
    # Extract text samples - CRITICAL: prefer raw text, avoid decoding with wrong tokenizer
    texts = []
    for i, example in enumerate(dataset):
        if i >= num_samples:
            break
        
        # PREFER raw text field if available (safest)
        if "text" in example:
            text = example["text"]
            if isinstance(text, list):
                # Handle list of strings (join if needed)
                text = " ".join(text) if text else ""
            texts.append(text)
            continue
        
        # If only input_ids available, we must decode
        # The tokenizer passed should be the one that created the cached dataset (model_tokenizer)
        if "input_ids" in example:
            # Only warn once, not for every example
            if i == 0:
                print(f"  ⚠ WARNING: Dataset contains 'input_ids' but no 'text' field.")
                print(f"     Decoding with tokenizer '{tokenizer.name_or_path}' (should match the tokenizer that created the cached dataset).")
                print(f"     If this tokenizer doesn't match the cached dataset's tokenizer, baselines will be incorrect.")
            
            if isinstance(example["input_ids"], torch.Tensor):
                text = tokenizer.decode(example["input_ids"].tolist(), skip_special_tokens=True, clean_up_tokenization_spaces=False)
            else:
                text = tokenizer.decode(example["input_ids"], skip_special_tokens=True, clean_up_tokenization_spaces=False)
            texts.append(text)
            continue
        
        # Fallback: try to find any text-like field
        for key, value in example.items():
            if key == "text" or (isinstance(value, str) and len(value) > 10):
                texts.append(str(value))
                break
        else:
            # No suitable field found, skip this example
            print(f"  ⚠ WARNING: Example {i} has no 'text' or 'input_ids' field, skipping")
            continue
    
    print(f"  Extracted {len(texts)} text samples")
    if len(texts) == 0:
        raise RuntimeError(
            f"No text samples extracted from dataset '{dataset_name}'. "
            f"Dataset may not contain 'text' field, or tokenizer mismatch caused decode failures."
        )
    return texts


def _retokenize(
    texts: List[str],
    tokenizer: AutoTokenizer,
    max_length: int,
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Re-tokenize texts with the reference tokenizer."""
    # Respect model's actual max length if available
    model_max_length = getattr(tokenizer, "model_max_length", None)
    if model_max_length is not None and model_max_length < max_length:
        effective_max_length = model_max_length
    else:
        effective_max_length = max_length
    
    batch = tokenizer(
        texts,
        return_tensors="pt",
        return_token_type_ids=False,
        return_attention_mask=True,
        truncation=True,
        padding=True,
        max_length=effective_max_length,
    )
    # Ensure attention_mask is long (0/1) not bool for HF model compatibility
    attn_mask = batch["attention_mask"].to(device).long()
    input_ids = batch["input_ids"].to(device)
    return input_ids, attn_mask


def _compute_boundary_error_nll_contributions(
    texts: List[str],
    token_ids: torch.Tensor,
    nll_tensor: torch.Tensor,
    valid_mask: torch.Tensor,
    tokenizer: AutoTokenizer,
) -> Dict[str, float]:
    """Compute NLL contributions from boundary errors (space/punctuation issues)."""
    # Flatten
    valid_nlls = nll_tensor[valid_mask.bool()].detach().cpu().numpy()
    valid_token_ids = token_ids[valid_mask.bool()].detach().cpu().numpy()
    
    if len(valid_nlls) == 0:
        return {}
    
    # Find tokens that are likely boundary errors
    boundary_error_nlls = []
    boundary_error_count = 0
    
    for i, (token_id, nll_val) in enumerate(zip(valid_token_ids, valid_nlls)):
        try:
            token_str = tokenizer.decode([int(token_id)], skip_special_tokens=False, clean_up_tokenization_spaces=False)
            # Check if this token is likely a boundary error
            # (space before punct, missing space after punct, double space)
            # This is approximate - we check the token string
            if any(c in token_str for c in ",.!?;:") and (" " in token_str or len(token_str.strip()) < len(token_str)):
                boundary_error_nlls.append(float(nll_val))
                boundary_error_count += 1
        except:
            pass
    
    if boundary_error_count == 0:
        return {
            "boundary_error_count": 0,
            "boundary_error_fraction": 0.0,
            "boundary_error_avg_nll": 0.0,
            "boundary_error_total_nll_contribution": 0.0,
        }
    
    total_nll = float(np.sum(valid_nlls))
    boundary_error_total_nll = float(np.sum(boundary_error_nlls))
    
    return {
        "boundary_error_count": boundary_error_count,
        "boundary_error_fraction": boundary_error_count / len(valid_nlls) if len(valid_nlls) > 0 else 0.0,
        "boundary_error_avg_nll": float(np.mean(boundary_error_nlls)),
        "boundary_error_total_nll_contribution": boundary_error_total_nll / total_nll if total_nll > 0 else 0.0,
    }


def _score_from_token_ids(
    token_ids: np.ndarray,
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    device: torch.device,
    batch_size: int,
    max_length: int,
    first_chunk_only: bool,
    ignore_prefix_tokens: int,
    prompt_len: Optional[int],
    top_k_worst_tokens: int = 50,
) -> Tuple[Dict[str, Any], List[WorstTokenInfo], List[Dict[str, Any]], torch.Tensor]:
    """Score directly from token IDs without decode/retokenize.
    
    CRITICAL FIXES:
    - Pad token handling: If pad_token_id == eos_token_id, treat samples as fixed-length (no padding detection)
    - prompt_len off-by-one: Label index j corresponds to original token index j+1, so prompt_len P maps to label index P-1
    - Attention mask dtype: Ensure long (0/1) not bool
    - Device safety: Ensure all tensors are on the correct device for the model
    
    Returns:
        (metrics, worst_tokens, per_sample_metrics, all_token_ids_tensor)
    """
    total_acc = 0.0
    total_nll = 0.0
    total_tokens = 0.0
    total_nll_full = 0.0
    total_tokens_full = 0.0
    all_nlls: List[float] = []
    
    worst_tokens_all: List[WorstTokenInfo] = []
    per_sample_metrics: List[Dict[str, Any]] = []
    
    nll_tensors_for_worst: List[torch.Tensor] = []
    input_ids_for_worst: List[torch.Tensor] = []  # Full input_ids for extract_worst_tokens
    labels_for_worst: List[torch.Tensor] = []  # Labels (shifted) for token-type functions
    valid_masks_for_worst: List[torch.Tensor] = []
    all_token_ids_list: List[torch.Tensor] = []
    
    # Get token IDs for device safety
    pad_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0
    eos_token_id = tokenizer.eos_token_id
    
    # CRITICAL: If pad_token_id == eos_token_id, we cannot distinguish padding from real EOS
    # In this case, treat all samples as fixed-length (no padding) and set attention_mask=1 everywhere
    pad_equals_eos = (pad_token_id == eos_token_id) if eos_token_id is not None else False
    
    with torch.no_grad():
        for batch_idx, i in enumerate(range(0, len(token_ids), batch_size)):
            batch_token_ids = token_ids[i : i + batch_size]
            
            # Convert to tensor and pad
            batch_tensors = []
            batch_masks = []
            for tokens in batch_token_ids:
                tokens_tensor = torch.tensor(tokens[:max_length], dtype=torch.long, device=device)
                
                # CRITICAL FIX: Pad token masking
                # If pad==eos, we cannot reliably detect padding, so treat as fixed-length (all valid)
                # Otherwise, exclude pad tokens from scoring
                if pad_equals_eos:
                    # Cannot distinguish pad from EOS, assume no padding (all tokens valid)
                    mask = torch.ones(len(tokens_tensor), dtype=torch.bool, device=device)
                else:
                    # Pad token is distinct from EOS, safe to exclude pad tokens
                    mask = (tokens_tensor != pad_token_id).to(torch.bool)
                
                batch_tensors.append(tokens_tensor)
                batch_masks.append(mask)
            
            # Pad to same length
            max_len = max(len(t) for t in batch_tensors) if batch_tensors else 1
            input_ids = torch.zeros(len(batch_tensors), max_len, dtype=torch.long, device=device)
            # CRITICAL FIX: Attention mask must be long (0/1) not bool for HF models
            attn_mask = torch.zeros(len(batch_tensors), max_len, dtype=torch.long, device=device)
            
            for j, (tokens_tensor, mask) in enumerate(zip(batch_tensors, batch_masks)):
                seq_len = len(tokens_tensor)
                input_ids[j, :seq_len] = tokens_tensor
                attn_mask[j, :seq_len] = mask.long()  # Ensure long dtype
            
            # Ensure input_ids and attn_mask are on the correct device for the model
            # Get device from model parameters
            model_device = next(model.parameters()).device
            input_ids = input_ids.to(model_device)
            attn_mask = attn_mask.to(model_device)
            
            # Compute logits
            logits = model(input_ids=input_ids, attention_mask=attn_mask, use_cache=False).logits[:, :-1]
            labels = input_ids[:, 1:]
            loss_mask = attn_mask[:, :-1].bool()  # Convert to bool for masking operations
            
            # Compute NLL
            nll = F.cross_entropy(logits.flatten(0, 1), labels.flatten(0, 1), reduction="none").view_as(labels)
            
            # Full sequence metrics
            valid_full = loss_mask.bool()
            total_nll_full += float((nll * valid_full.to(nll.dtype)).sum().item())
            total_tokens_full += float(valid_full.sum().item())
            
            # Apply chunking and prefix handling
            if first_chunk_only:
                eos_id = tokenizer.eos_token_id
                if eos_id is not None:
                    eos_mask = (labels == eos_id).cumsum(-1) == 0
                else:
                    eos_mask = torch.ones_like(labels, dtype=torch.bool)
                valid = loss_mask.bool() & eos_mask
            else:
                valid = loss_mask.bool()
            
            # CRITICAL FIX: prompt_len off-by-one correction
            # Label index j corresponds to original token index j+1
            # If prompt has P tokens (original indices 0..P-1), we want to score from original index P onwards
            # This corresponds to label index P-1 (since labels start at original index 1)
            if prompt_len is not None and prompt_len > 0:
                # Map prompt_len (original token space) to label space: label_start = max(prompt_len - 1, 0)
                label_start = max(prompt_len - 1, 0)
                suffix_mask = torch.arange(valid.shape[1], device=valid.device) >= label_start
                valid = valid & suffix_mask.unsqueeze(0)
            
            # ignore_prefix_tokens is already in label space (applied after shift), so no correction needed
            if ignore_prefix_tokens > 0:
                prefix_mask = torch.arange(valid.shape[1], device=valid.device) >= ignore_prefix_tokens
                valid = valid & prefix_mask.unsqueeze(0)
            
            valid = valid.to(nll.dtype)
            batch_nlls = nll[valid == 1].detach().cpu().numpy().tolist()
            all_nlls.extend(batch_nlls)
            total_nll += float((nll * valid).sum().item())
            
            acc = (logits.argmax(-1) == labels).to(nll.dtype)
            total_acc += float((acc * valid).sum().item())
            total_tokens += float(valid.sum().item())
            
            # Store for worst token extraction
            if batch_idx < 10:
                nll_tensors_for_worst.append(nll.detach().cpu())
                # Store full input_ids for extract_worst_tokens (needs [B, L])
                input_ids_for_worst.append(input_ids.detach().cpu())
                # Store labels (input_ids[:, 1:]) for token-type functions (needs [B, L-1])
                labels_for_worst.append(labels.detach().cpu())
                valid_masks_for_worst.append(valid.detach().cpu())
                all_token_ids_list.append(input_ids.detach().cpu())  # Keep full sequence for other uses
            
            # Per-sample metrics with tail diagnosis
            for sample_idx_in_batch in range(len(batch_token_ids)):
                sample_valid = valid[sample_idx_in_batch]
                sample_nll = nll[sample_idx_in_batch]
                sample_labels = labels[sample_idx_in_batch]
                sample_nlls = sample_nll[sample_valid.bool()].detach().cpu().numpy().tolist()
                sample_token_ids = sample_labels[sample_valid.bool()].detach().cpu().numpy().tolist()
                
                if len(sample_nlls) > 0:
                    # Compute quarter stats
                    quarter_stats = _compute_quarter_stats(sample_nlls, sample_token_ids, tokenizer)
                    
                    per_sample_metrics.append({
                        "sample_idx": i + sample_idx_in_batch,
                        "length_tokens": int(sample_valid.sum().item()),
                        "avg_nll": float(np.mean(sample_nlls)),
                        "median_nll": float(np.median(sample_nlls)),
                        "max_nll": float(np.max(sample_nlls)) if sample_nlls else 0.0,
                        "p99_nll": float(np.percentile(sample_nlls, 99)) if len(sample_nlls) > 0 else 0.0,
                        "quarter_stats": quarter_stats,
                    })
    
    if total_tokens == 0:
        raise RuntimeError("No valid tokens for evaluation")
    
    # Extract worst tokens - CRITICAL FIX: Decode properly for token-ID pathway
    worst_tokens_with_flags = []
    if nll_tensors_for_worst:
        all_nll_tensor = torch.cat(nll_tensors_for_worst, dim=0)
        all_input_ids = torch.cat(input_ids_for_worst, dim=0)
        all_valid_mask = torch.cat(valid_masks_for_worst, dim=0)
        
        # For token-ID pathway, decode input_ids to get proper context texts
        # Use the same tokenizer (eval_tokenizer) that was used for scoring
        decoded_texts = []
        for sample_input_ids in all_input_ids:
            # Decode the full sequence for context extraction
            text = tokenizer.decode(sample_input_ids.tolist(), skip_special_tokens=False, clean_up_tokenization_spaces=False)
            decoded_texts.append(text)
        
        worst_tokens_all = extract_worst_tokens(
            all_nll_tensor,
            all_input_ids,
            all_valid_mask,
            decoded_texts,  # Use decoded texts instead of dummy texts
            tokenizer,
            top_k=top_k_worst_tokens,
            context_chars=40,
            batch_offset=0,
        )
        
        # Extract worst tokens with artifact flags and token context
        worst_tokens_with_flags = _extract_worst_tokens_with_flags(
            all_nll_tensor,
            all_input_ids,
            all_valid_mask,
            tokenizer,
            top_k=top_k_worst_tokens,
            context_tokens=10,
            batch_offset=0,
        )
    
    # Compute metrics
    avg_nll = total_nll / total_tokens
    ppl = float(np.exp(avg_nll))
    acc = total_acc / total_tokens
    avg_nll_full = total_nll_full / total_tokens_full if total_tokens_full > 0 else float("nan")
    ppl_full = float(np.exp(avg_nll_full)) if not np.isnan(avg_nll_full) else float("nan")
    
    # Tail statistics
    tail_stats = compute_nll_tail_stats(all_nlls)
    
    # Robust metrics
    robust_metrics = compute_robust_metrics(all_nlls)
    
    # Tail contribution
    tail_contribution = compute_tail_contribution(all_nlls)
    
    # Token-type stratified NLL (if we have token IDs)
    # Use labels (shifted) to match nll_tensor and valid_mask shapes
    all_labels_tensor = torch.cat(labels_for_worst, dim=0) if labels_for_worst else None
    all_token_ids_tensor = torch.cat(all_token_ids_list, dim=0) if all_token_ids_list else None
    token_type_stats = {}
    boundary_error_stats = {}
    if all_labels_tensor is not None and nll_tensors_for_worst:
        all_nll_tensor = torch.cat(nll_tensors_for_worst, dim=0)
        all_valid_mask = torch.cat(valid_masks_for_worst, dim=0)
        token_type_stats = compute_token_type_stratified_nll(
            all_labels_tensor,
            all_nll_tensor,
            all_valid_mask,
            tokenizer,
        )
        # Compute boundary error NLL contributions
        # For token-ID pathway, we don't have original texts, so use empty list
        # (boundary error detection relies on token strings, which we can decode)
        dummy_texts = [""] * all_labels_tensor.shape[0]  # Not used for token-ID scoring
        boundary_error_stats = _compute_boundary_error_nll_contributions(
            dummy_texts,
            all_labels_tensor,
            all_nll_tensor,
            all_valid_mask,
            tokenizer,
        )
    
    # Compute tail diagnosis
    tail_diagnosis = {
        "worst_tokens": worst_tokens_with_flags,
    }
    
    metrics = {
        "avg_nll": float(avg_nll),
        "median_nll": float(np.median(all_nlls)) if all_nlls else float("nan"),
        "ppl": float(ppl),
        "acc": float(acc),
        "tokens_scored": int(total_tokens),
        "tokens_total": int(total_tokens_full),
        "avg_nll_full": float(avg_nll_full) if not np.isnan(avg_nll_full) else None,
        "ppl_full": float(ppl_full) if not np.isnan(ppl_full) else float("nan"),
        "nll_delta_full_vs_scored": float(avg_nll_full - avg_nll) if not np.isnan(avg_nll_full) else None,
        "tail_stats": tail_stats,
        "robust_metrics": robust_metrics,
        "tail_contribution": tail_contribution,
        "token_type_stats": token_type_stats,
        "boundary_error_stats": boundary_error_stats,
        "tail_diagnosis": tail_diagnosis,
    }
    
    return metrics, worst_tokens_all, per_sample_metrics, all_token_ids_tensor


def _score_with_reference_model(
    texts: List[str],
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    device: torch.device,
    batch_size: int,
    max_length: int,
    retokenize: bool,
    first_chunk_only: bool,
    ignore_prefix_tokens: int,
    prompt_len: Optional[int],
    top_k_worst_tokens: int = 50,
) -> Tuple[Dict[str, Any], List[WorstTokenInfo], List[Dict[str, Any]]]:
    """Score texts with a reference model and return comprehensive metrics.
    
    CRITICAL FIXES:
    - prompt_len off-by-one: Label index j corresponds to original token index j+1
    - Attention mask dtype: Ensure long (0/1) not bool
    - Device safety: Ensure all tensors are on the correct device for the model
    - Remove dead code: retokenize parameter is always True in practice, but keep for API consistency
    """
    total_acc = 0.0
    total_nll = 0.0
    total_tokens = 0.0
    total_nll_full = 0.0
    total_tokens_full = 0.0
    all_nlls: List[float] = []
    
    # Store per-token NLL for tail analysis (memory-efficient: only store what we need)
    worst_tokens_all: List[WorstTokenInfo] = []
    per_sample_metrics: List[Dict[str, Any]] = []
    
    # Store NLL tensors for worst token extraction (only for first batch to save memory)
    nll_tensors_for_worst: List[torch.Tensor] = []
    input_ids_for_worst: List[torch.Tensor] = []  # Full input_ids for extract_worst_tokens
    labels_for_worst: List[torch.Tensor] = []  # Labels (shifted) for token-type functions
    valid_masks_for_worst: List[torch.Tensor] = []
    texts_for_worst: List[str] = []
    
    # Get device from model parameters (handles device_map="auto" case)
    model_device = next(model.parameters()).device
    
    with torch.no_grad():
        for batch_idx, i in enumerate(range(0, len(texts), batch_size)):
            batch_texts = texts[i : i + batch_size]
            
            # Always retokenize in text-based pathway (retokenize parameter kept for API consistency)
            input_ids, attn_mask = _retokenize(batch_texts, tokenizer, max_length, device)
            
            # Ensure tensors are on the correct device for the model
            input_ids = input_ids.to(model_device)
            attn_mask = attn_mask.to(model_device)
            
            # Compute logits
            logits = model(input_ids=input_ids, attention_mask=attn_mask, use_cache=False).logits[:, :-1]
            labels = input_ids[:, 1:]
            loss_mask = attn_mask[:, :-1].bool()  # Convert to bool for masking operations
            
            # Compute NLL
            nll = F.cross_entropy(logits.flatten(0, 1), labels.flatten(0, 1), reduction="none").view_as(labels)
            
            # Full sequence metrics
            valid_full = loss_mask.bool()
            total_nll_full += float((nll * valid_full.to(nll.dtype)).sum().item())
            total_tokens_full += float(valid_full.sum().item())
            
            # Apply chunking and prefix handling
            if first_chunk_only:
                eos_id = tokenizer.eos_token_id
                if eos_id is not None:
                    eos_mask = (labels == eos_id).cumsum(-1) == 0
                else:
                    eos_mask = torch.ones_like(labels, dtype=torch.bool)
                valid = loss_mask.bool() & eos_mask
            else:
                valid = loss_mask.bool()
            
            # CRITICAL FIX: prompt_len off-by-one correction
            # Label index j corresponds to original token index j+1
            # If prompt has P tokens (original indices 0..P-1), we want to score from original index P onwards
            # This corresponds to label index P-1 (since labels start at original index 1)
            if prompt_len is not None and prompt_len > 0:
                # Map prompt_len (original token space) to label space: label_start = max(prompt_len - 1, 0)
                label_start = max(prompt_len - 1, 0)
                suffix_mask = torch.arange(valid.shape[1], device=valid.device) >= label_start
                valid = valid & suffix_mask.unsqueeze(0)
            
            # ignore_prefix_tokens is already in label space (applied after shift), so no correction needed
            if ignore_prefix_tokens > 0:
                prefix_mask = torch.arange(valid.shape[1], device=valid.device) >= ignore_prefix_tokens
                valid = valid & prefix_mask.unsqueeze(0)
            
            valid = valid.to(nll.dtype)
            batch_nlls = nll[valid == 1].detach().cpu().numpy().tolist()
            all_nlls.extend(batch_nlls)
            total_nll += float((nll * valid).sum().item())
            
            acc = (logits.argmax(-1) == labels).to(nll.dtype)
            total_acc += float((acc * valid).sum().item())
            total_tokens += float(valid.sum().item())
            
            # Store for worst token extraction (only first few batches to save memory)
            if batch_idx < 10:  # Limit memory usage
                nll_tensors_for_worst.append(nll.detach().cpu())
                # Store full input_ids for extract_worst_tokens (needs [B, L])
                input_ids_for_worst.append(input_ids.detach().cpu())
                # Store labels (input_ids[:, 1:]) for token-type functions (needs [B, L-1])
                labels_for_worst.append(labels.detach().cpu())
                valid_masks_for_worst.append(valid.detach().cpu())
                texts_for_worst.extend(batch_texts)
            
            # Per-sample metrics (for all batches) with tail diagnosis
            for sample_idx_in_batch in range(len(batch_texts)):
                sample_valid = valid[sample_idx_in_batch]
                sample_nll = nll[sample_idx_in_batch]
                sample_labels = labels[sample_idx_in_batch]
                sample_nlls = sample_nll[sample_valid.bool()].detach().cpu().numpy().tolist()
                sample_token_ids = sample_labels[sample_valid.bool()].detach().cpu().numpy().tolist()
                
                if len(sample_nlls) > 0:
                    # Compute quarter stats
                    quarter_stats = _compute_quarter_stats(sample_nlls, sample_token_ids, tokenizer)
                    
                    per_sample_metrics.append({
                        "sample_idx": i + sample_idx_in_batch,
                        "length_chars": len(batch_texts[sample_idx_in_batch]),
                        "length_words": len(batch_texts[sample_idx_in_batch].split()),
                        "length_tokens": int(sample_valid.sum().item()),
                        "avg_nll": float(np.mean(sample_nlls)),
                        "median_nll": float(np.median(sample_nlls)),
                        "max_nll": float(np.max(sample_nlls)) if sample_nlls else 0.0,
                        "p99_nll": float(np.percentile(sample_nlls, 99)) if len(sample_nlls) > 0 else 0.0,
                        "quarter_stats": quarter_stats,
                    })
            
    
    if total_tokens == 0:
        raise RuntimeError("No valid tokens for evaluation")
    
    # Extract worst tokens from stored batches (combine across batches)
    worst_tokens_with_flags = []
    if nll_tensors_for_worst:
        # Combine batches
        all_nll_tensor = torch.cat(nll_tensors_for_worst, dim=0)
        all_input_ids = torch.cat(input_ids_for_worst, dim=0)
        all_valid_mask = torch.cat(valid_masks_for_worst, dim=0)
        
        worst_tokens_all = extract_worst_tokens(
            all_nll_tensor,
            all_input_ids,
            all_valid_mask,
            texts_for_worst,
            tokenizer,
            top_k=top_k_worst_tokens,
            context_chars=40,
            batch_offset=0,  # These are already from first batches
        )
        
        # Extract worst tokens with artifact flags and token context
        worst_tokens_with_flags = _extract_worst_tokens_with_flags(
            all_nll_tensor,
            all_input_ids,
            all_valid_mask,
            tokenizer,
            top_k=top_k_worst_tokens,
            context_tokens=10,
            batch_offset=0,
        )
    
    # Compute metrics
    avg_nll = total_nll / total_tokens
    ppl = float(np.exp(avg_nll))
    acc = total_acc / total_tokens
    avg_nll_full = total_nll_full / total_tokens_full if total_tokens_full > 0 else float("nan")
    ppl_full = float(np.exp(avg_nll_full)) if not np.isnan(avg_nll_full) else float("nan")
    
    # Tail statistics
    tail_stats = compute_nll_tail_stats(all_nlls)
    
    # Robust metrics
    robust_metrics = compute_robust_metrics(all_nlls)
    
    # Tail contribution
    tail_contribution = compute_tail_contribution(all_nlls)
    
    # Token-type stratified NLL and boundary error NLL contributions (for text-based pathway)
    token_type_stats = {}
    boundary_error_stats = {}
    if nll_tensors_for_worst and labels_for_worst and valid_masks_for_worst:
        all_nll_tensor = torch.cat(nll_tensors_for_worst, dim=0)
        all_labels = torch.cat(labels_for_worst, dim=0)  # Use labels (shifted) for token-type functions
        all_valid_mask = torch.cat(valid_masks_for_worst, dim=0)
        
        # Token-type stratified NLL
        token_type_stats = compute_token_type_stratified_nll(
            all_labels,
            all_nll_tensor,
            all_valid_mask,
            tokenizer,
        )
        
        # Boundary error NLL contributions
        boundary_error_stats = _compute_boundary_error_nll_contributions(
            texts_for_worst,
            all_labels,
            all_nll_tensor,
            all_valid_mask,
            tokenizer,
        )
    
    # Compute tail diagnosis
    tail_diagnosis = {
        "worst_tokens": worst_tokens_with_flags,
    }
    
    metrics = {
        "avg_nll": float(avg_nll),
        "median_nll": float(np.median(all_nlls)) if all_nlls else float("nan"),
        "ppl": float(ppl),
        "acc": float(acc),
        "tokens_scored": int(total_tokens),
        "tokens_total": int(total_tokens_full),
        "avg_nll_full": float(avg_nll_full) if not np.isnan(avg_nll_full) else None,
        "ppl_full": float(ppl_full) if not np.isnan(ppl_full) else float("nan"),
        "nll_delta_full_vs_scored": float(avg_nll_full - avg_nll) if not np.isnan(avg_nll_full) else None,
        "tail_stats": tail_stats,
        "robust_metrics": robust_metrics,
        "tail_contribution": tail_contribution,
        "token_type_stats": token_type_stats,
        "boundary_error_stats": boundary_error_stats,
        "tail_diagnosis": tail_diagnosis,
    }
    
    return metrics, worst_tokens_all, per_sample_metrics


@hydra.main(config_path="../../../configs/eval", config_name="gen_ppl", version_base="1.3")
def main(cfg: DictConfig):
    """Main evaluation function."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.set_float32_matmul_precision("high")
    torch.set_grad_enabled(False)
    
    print("=" * 80)
    print("Generative Perplexity Evaluation")
    print("Cross-PPL under Reference Language Models")
    print("=" * 80)
    print()
    
    # Parse reference models (support comma-separated list)
    if hasattr(cfg, "reference_models") and cfg.reference_models:
        reference_models = [m.strip() for m in str(cfg.reference_models).split(",")]
    elif hasattr(cfg, "reference_model") and cfg.reference_model:
        reference_models = [cfg.reference_model]
    else:
        reference_models = ["gpt2-large"]
    
    print(f"Reference model(s): {', '.join(reference_models)}")
    
    # Load model tokenizer
    print(f"Loading model tokenizer: {cfg.model_tokenizer}")
    model_tokenizer = AutoTokenizer.from_pretrained(cfg.model_tokenizer)
    if model_tokenizer.pad_token_id is None:
        model_tokenizer.pad_token = model_tokenizer.eos_token
    # Wrap decode methods to always include clean_up_tokenization_spaces=False
    wrap_tokenizer_decode_methods(model_tokenizer)
    
    # Load samples or real text
    if hasattr(cfg, "real_text_dataset") and cfg.real_text_dataset and cfg.real_text_dataset != "null":
        print(f"\nLoading real text from dataset: {cfg.real_text_dataset}")
        num_real_samples = cfg.get("real_text_num_samples", 1000)
        # For real text baseline, use model_tokenizer (the one that created the cached dataset)
        # This ensures correct decoding if we need to decode input_ids
        real_tokenizer = model_tokenizer  # Use the same tokenizer that created the cached dataset
        
        texts = _load_real_text_samples(
            cfg.real_text_dataset,
            num_real_samples,
            real_tokenizer,  # Only used if dataset has input_ids but no text field
            cache_dir=cfg.get("real_text_cache_dir", None),
        )
        print(f"Loaded {len(texts)} real text samples")
        is_real_text = True
        original_token_ids = None  # Real text baseline - no original token IDs available
    else:
        if not hasattr(cfg, "samples_path") or not cfg.samples_path:
            raise ValueError("Either samples_path or real_text_dataset must be provided")
        print(f"\nLoading samples from: {cfg.samples_path}")
        z_ts = _load_samples(cfg.samples_path)
        if z_ts.ndim != 2:
            raise ValueError(f"Expected 2D [N, T] tokens array, got {z_ts.shape}")
        print(f"Loaded {len(z_ts)} samples with shape {z_ts.shape}")
        
        # Diagnostic: Check for pad tokens and special tokens
        pad_token_id = model_tokenizer.pad_token_id if model_tokenizer.pad_token_id is not None else 0
        eos_token_id = model_tokenizer.eos_token_id
        unk_token_id = model_tokenizer.unk_token_id
        
        # Count pad tokens in samples
        pad_counts = np.sum(z_ts == pad_token_id, axis=1)
        eos_counts = np.sum(z_ts == eos_token_id, axis=1) if eos_token_id is not None else np.zeros(len(z_ts))
        unk_counts = np.sum(z_ts == unk_token_id, axis=1) if unk_token_id is not None else np.zeros(len(z_ts))
        
        print(f"  Sample diagnostics:")
        print(f"    Pad tokens: avg={np.mean(pad_counts):.1f}, max={np.max(pad_counts)}, samples_with_pad={np.sum(pad_counts > 0)}/{len(z_ts)}")
        if eos_token_id is not None:
            print(f"    EOS tokens: avg={np.mean(eos_counts):.1f}, max={np.max(eos_counts)}, samples_with_eos={np.sum(eos_counts > 0)}/{len(z_ts)}")
        if unk_token_id is not None:
            print(f"    UNK tokens: avg={np.mean(unk_counts):.1f}, max={np.max(unk_counts)}, samples_with_unk={np.sum(unk_counts > 0)}/{len(z_ts)}")
        
        # Decode with configurable options
        decode_skip_special = cfg.get("decode_skip_special_tokens", False)
        decode_cleanup = cfg.get("decode_cleanup_spaces", False)
        texts = model_tokenizer.batch_decode(
            z_ts,
            skip_special_tokens=decode_skip_special,
            clean_up_tokenization_spaces=decode_cleanup,
        )
        is_real_text = False
        original_token_ids = z_ts  # Keep original token IDs for direct scoring
    
    # =========================================================================
    # Token-ID Degeneracy Diagnostics (computed once, not per reference model)
    # Only for generated samples, not real-text baseline
    # =========================================================================
    token_id_degeneracy = None
    if original_token_ids is not None:
        print("\nComputing token-ID degeneracy metrics...")
        # Support multiple n-gram orders for degeneracy diagnostics.
        # Default: use [2, 3, 4] if neither deg_ngram_ns nor deg_ngram_n is set.
        deg_ngram_ns_cfg = cfg.get("deg_ngram_ns", None)
        if deg_ngram_ns_cfg is not None:
            deg_ngram_ns = deg_ngram_ns_cfg
        else:
            # Backward compatibility: fall back to single n if explicitly provided,
            # otherwise use the new default [2, 3, 4].
            deg_ngram_n = cfg.get("deg_ngram_n", None)
            if deg_ngram_n is None:
                deg_ngram_ns = [2, 3, 4]
            else:
                deg_ngram_ns = [deg_ngram_n]
        # Normalize to a list of ints
        if isinstance(deg_ngram_ns, (int, float)):
            deg_ngram_ns = [int(deg_ngram_ns)]
        else:
            deg_ngram_ns = [int(n) for n in deg_ngram_ns]

        deg_pair_sample = cfg.get("deg_pair_sample", 2000)
        deg_seed = cfg.get("deg_seed", 0)

        token_id_degeneracy = {}
        for n in deg_ngram_ns:
            token_id_degeneracy[f"n{n}"] = compute_token_id_degeneracy_metrics(
                token_ids=original_token_ids,
                ngram_n=n,
                pair_sample=deg_pair_sample,
                seed=deg_seed,
            )

        # Print unique/duplicate once (independent of n-gram order)
        # Use the first entry as representative
        first_key = next(iter(token_id_degeneracy))
        base_deg = token_id_degeneracy[first_key]
        print(f"  Unique sequence ratio: {base_deg['token_id_unique_ratio']:.4f}")
        print(f"  Duplicate rate: {base_deg['token_id_duplicate_rate']:.4f}")
        # Then print per-n diagnostics
        for n in deg_ngram_ns:
            deg = token_id_degeneracy[f"n{n}"]
            print(f"  Repeated {n}-gram frac (mean): {deg['rep_ngram_frac_mean']:.4f}")
            print(f"  Repeated {n}-gram frac (p95): {deg['rep_ngram_frac_p95']:.4f}")
            print(f"  Inter-sample Jaccard mean ({deg['pair_sample']} pairs, n={n}): {deg['inter_sample_jaccard_mean']:.4f}")
    
    # Score with each reference model
    all_results = {}
    worst_tokens_by_model = {}  # Store worst tokens separately (not JSON-serializable, only for generated samples)
    
    for ref_model_name in reference_models:
        print(f"\n{'=' * 80}")
        print(f"Scoring with reference model: {ref_model_name}")
        print("=" * 80)
        
        # CRITICAL FIX: Device handling - remove device_map="auto" for deterministic device placement
        # Load model and move to device explicitly to avoid device_map sharding issues
        eval_model = AutoModelForCausalLM.from_pretrained(ref_model_name)
        eval_model = eval_model.to(device)
        eval_model.eval()  # Set to eval mode
        
        eval_tokenizer = AutoTokenizer.from_pretrained(ref_model_name)
        if eval_tokenizer.pad_token_id is None:
            eval_tokenizer.pad_token = eval_tokenizer.eos_token
        # Wrap decode methods to always include clean_up_tokenization_spaces=False
        wrap_tokenizer_decode_methods(eval_tokenizer)
        
        # CRITICAL FIX: torch.compile only when safe (model is on single device)
        if cfg.torch_compile:
            print("Compiling model...")
            eval_model = torch.compile(eval_model)
        
        # Check tokenizer compatibility for direct token-ID scoring
        token_id_scoring_enabled = False
        token_id_metrics = None
        token_id_per_sample_metrics = None
        
        if not is_real_text and original_token_ids is not None and not cfg.retokenize:
            print("Checking tokenizer compatibility for direct token-ID scoring...")
            is_compatible, reason = check_tokenizer_compatibility(model_tokenizer, eval_tokenizer)
            print(f"  Compatibility check: {is_compatible} ({reason})")
            
            # CRITICAL FIX: Additional strict compatibility checks
            if is_compatible:
                # Additional consistency checks
                additional_checks_passed = True
                check_messages = []
                
                # Check vocab size
                if model_tokenizer.vocab_size != eval_tokenizer.vocab_size:
                    additional_checks_passed = False
                    check_messages.append(f"vocab_size mismatch: {model_tokenizer.vocab_size} vs {eval_tokenizer.vocab_size}")
                
                # Check special token IDs
                special_token_checks = [
                    ("eos_token_id", model_tokenizer.eos_token_id, eval_tokenizer.eos_token_id),
                    ("bos_token_id", getattr(model_tokenizer, "bos_token_id", None), getattr(eval_tokenizer, "bos_token_id", None)),
                    ("unk_token_id", model_tokenizer.unk_token_id, eval_tokenizer.unk_token_id),
                    ("pad_token_id", model_tokenizer.pad_token_id, eval_tokenizer.pad_token_id),
                ]
                for name, model_val, eval_val in special_token_checks:
                    if model_val != eval_val:
                        additional_checks_passed = False
                        check_messages.append(f"{name} mismatch: {model_val} vs {eval_val}")
                
                # Check tokenizer class name
                model_class = type(model_tokenizer).__name__
                eval_class = type(eval_tokenizer).__name__
                if model_class != eval_class:
                    additional_checks_passed = False
                    check_messages.append(f"tokenizer class mismatch: {model_class} vs {eval_class}")
                
                if not additional_checks_passed:
                    is_compatible = False
                    reason = f"Additional checks failed: {', '.join(check_messages)}"
                    print(f"  ⚠ Compatibility rejected: {reason}")
                    print(f"  Falling back to text-based scoring for safety")
                else:
                    print(f"  ✓ All compatibility checks passed")
            
            if is_compatible:
                print("  Using direct token-ID scoring pathway (no decode/retokenize)")
                token_id_scoring_enabled = True
                token_id_metrics, _, token_id_per_sample_metrics, _ = _score_from_token_ids(
                    token_ids=original_token_ids,
                    model=eval_model,
                    tokenizer=eval_tokenizer,
                    device=device,
                    batch_size=cfg.batch_size,
                    max_length=cfg.max_length,
                    first_chunk_only=cfg.first_chunk_only,
                    ignore_prefix_tokens=cfg.get("ignore_prefix_tokens", 0),
                    prompt_len=cfg.get("prompt_len", None),
                    top_k_worst_tokens=cfg.get("top_k_worst_tokens", 50),
                )
                print(f"  Token-ID PPL: {token_id_metrics['ppl']:.2f}")
            else:
                print(f"  Tokenizers not compatible ({reason}), falling back to decode+retokenize")
        
        # Score with text-based pathway (always compute for comparison)
        if cfg.retokenize or not token_id_scoring_enabled:
            print("Using text-based scoring pathway (decode+retokenize)")
            text_metrics, worst_tokens, per_sample_metrics = _score_with_reference_model(
                texts=texts,
                model=eval_model,
                tokenizer=eval_tokenizer,
                device=device,
                batch_size=cfg.batch_size,
                max_length=cfg.max_length,
                retokenize=True,  # Always retokenize in text pathway
                first_chunk_only=cfg.first_chunk_only,
                ignore_prefix_tokens=cfg.get("ignore_prefix_tokens", 0),
                prompt_len=cfg.get("prompt_len", None),
                top_k_worst_tokens=cfg.get("top_k_worst_tokens", 50),
            )
            print(f"  Text-based PPL: {text_metrics['ppl']:.2f}")
        else:
            # Use token-ID metrics as primary, but still compute text metrics for comparison
            print("Computing text-based metrics for comparison...")
            text_metrics, worst_tokens, per_sample_metrics = _score_with_reference_model(
                texts=texts,
                model=eval_model,
                tokenizer=eval_tokenizer,
                device=device,
                batch_size=cfg.batch_size,
                max_length=cfg.max_length,
                retokenize=True,
                first_chunk_only=cfg.first_chunk_only,
                ignore_prefix_tokens=cfg.get("ignore_prefix_tokens", 0),
                prompt_len=cfg.get("prompt_len", None),
                top_k_worst_tokens=cfg.get("top_k_worst_tokens", 50),
            )
            print(f"  Text-based PPL: {text_metrics['ppl']:.2f}")
        
        # Use token-ID metrics as primary if available, otherwise use text metrics
        if token_id_scoring_enabled:
            metrics = token_id_metrics.copy()
            metrics["scoring_mode"] = "token_id_direct"
            metrics["text_based_comparison"] = {
                "ppl": text_metrics["ppl"],
                "avg_nll": text_metrics["avg_nll"],
                "ppl_delta": text_metrics["ppl"] - token_id_metrics["ppl"],
                "nll_delta": text_metrics["avg_nll"] - token_id_metrics["avg_nll"],
            }
            per_sample_metrics = token_id_per_sample_metrics
        else:
            metrics = text_metrics.copy()
            metrics["scoring_mode"] = "text_based"
        
        # Compute additional metrics
        print("Computing artifact metrics...")
        artifact_metrics = compute_artifact_metrics(texts, eval_tokenizer)
        
        print("Computing diversity metrics...")
        diversity_metrics = compute_enhanced_diversity_metrics(texts, eval_tokenizer)
        
        print("Computing entropy metrics...")
        entropy_metrics = compute_entropy_metrics(texts, eval_tokenizer)
        
        # CRITICAL FIX: Remove duplicate computation blocks
        # Byte-level validity diagnostics (compute once)
        byte_validity = {}
        if not is_real_text and original_token_ids is not None:
            print("Computing byte-level validity diagnostics...")
            byte_validity = compute_byte_level_validity(
                texts,
                original_token_ids,
                eval_tokenizer,
                num_check=cfg.get("num_roundtrip_check", 100),
            )
        
        # Bootstrap confidence intervals (compute once)
        bootstrap_cis = {}
        if per_sample_metrics and cfg.get("compute_bootstrap_ci", True):
            print("Computing bootstrap confidence intervals...")
            bootstrap_cis = compute_bootstrap_ci(
                per_sample_metrics,
                metric_key="avg_nll",
                n_bootstrap=cfg.get("bootstrap_n_samples", 1000),
                confidence=cfg.get("bootstrap_confidence", 0.95),
                seed=cfg.get("bootstrap_seed", 42),
            )
        
        # Add longest repeated substring
        longest_repeats = [compute_longest_repeated_substring(text) for text in texts]
        diversity_metrics["avg_longest_repeated_substring"] = float(np.mean(longest_repeats)) if longest_repeats else 0.0
        diversity_metrics["max_longest_repeated_substring"] = int(np.max(longest_repeats)) if longest_repeats else 0
        
        # Optional: MAUVE or embedding distance
        distributional_metrics = {}
        if cfg.get("compute_mauve", False) and not is_real_text:
            print("Computing MAUVE score (this may take a while)...")
            # Use a subset of real text as reference
            try:
                ref_cache_dir = _default_dataset_cache_dir("openwebtext-valid")
                
                # Load reference text from pre-cached validation dataset
                # Use model_tokenizer (the one that created the cached dataset) to decode correctly
                ref_texts = _load_real_text_samples(
                    cfg.get("mauve_reference_dataset", "openwebtext-valid"),
                    min(1000, len(texts)),
                    model_tokenizer,  # Use model_tokenizer, not eval_tokenizer, to match cached dataset
                    cache_dir=ref_cache_dir,
                )
                mauve_result = compute_mauve_score(texts[:min(1000, len(texts))], ref_texts)
                if mauve_result:
                    distributional_metrics["mauve"] = mauve_result
            except Exception as e:
                error_msg = str(e)
                if "Offline mode" in error_msg or "Hugging Face Hub" in error_msg or "Failed to load dataset" in error_msg:
                    print(f"  ⚠ MAUVE skipped: Validation dataset not cached. Run training script to cache it.")
                else:
                    print(f"  ⚠ MAUVE computation failed: {error_msg}")
        
        if cfg.get("compute_embedding_distance", False) and not is_real_text:
            print("Computing embedding distance...")
            try:
                ref_cache_dir = _default_dataset_cache_dir("openwebtext-valid")
                
                # Load reference text from pre-cached validation dataset
                # Use model_tokenizer (the one that created the cached dataset) to decode correctly
                ref_texts = _load_real_text_samples(
                    cfg.get("embedding_reference_dataset", "openwebtext-valid"),
                    min(1000, len(texts)),
                    model_tokenizer,  # Use model_tokenizer, not eval_tokenizer, to match cached dataset
                    cache_dir=ref_cache_dir,
                )
                embedding_result = compute_embedding_distance(texts[:min(1000, len(texts))], ref_texts)
                if embedding_result:
                    distributional_metrics["embedding"] = embedding_result
            except Exception as e:
                error_msg = str(e)
                if "Offline mode" in error_msg or "Hugging Face Hub" in error_msg or "Failed to load dataset" in error_msg:
                    print(f"  ⚠ Embedding distance skipped: Validation dataset not cached. Run training script to cache it.")
                else:
                    print(f"  ⚠ Embedding distance computation failed: {error_msg}")
        
        # Convert worst tokens to dict for JSON serialization
        worst_tokens_dict = [
            {
                "sample_idx": wt.sample_idx,
                "token_idx": wt.token_idx,
                "token_id": wt.token_id,
                "token_str": wt.token_str,
                "nll": wt.nll,
                "context_before": wt.context_before,
                "context_after": wt.context_after,
            }
            for wt in worst_tokens
        ]
        
        # Artifact sensitivity check (optional)
        artifact_sensitivity = {}
        if cfg.get("check_artifact_sensitivity", False) and not is_real_text:
            print("Computing artifact sensitivity check...")
            original_ppl = metrics.get("ppl", float("nan"))
            sanitized_result = _check_artifact_sensitivity(
                texts=texts,
                tokenizer=eval_tokenizer,
                model=eval_model,
                device=device,
                batch_size=cfg.batch_size,
                max_length=cfg.max_length,
                first_chunk_only=cfg.first_chunk_only,
                ignore_prefix_tokens=cfg.get("ignore_prefix_tokens", 0),
                prompt_len=cfg.get("prompt_len", None),
            )
            sanitized_ppl = sanitized_result.get("sanitized_ppl", None)
            if sanitized_ppl is not None and not np.isnan(original_ppl):
                ppl_delta = sanitized_ppl - original_ppl
                ppl_delta_pct = (ppl_delta / original_ppl) * 100 if original_ppl > 0 else 0.0
                artifact_sensitivity = {
                    "original_ppl": float(original_ppl),
                    "sanitized_ppl": float(sanitized_ppl),
                    "ppl_delta": float(ppl_delta),
                    "ppl_delta_pct": float(ppl_delta_pct),
                    "is_artifact_driven": abs(ppl_delta_pct) > 10.0,  # Label as artifact-driven if delta > 10%
                }
                print(f"  Original PPL: {original_ppl:.2f}")
                print(f"  Sanitized PPL: {sanitized_ppl:.2f}")
                print(f"  Delta: {ppl_delta:.2f} ({ppl_delta_pct:.1f}%)")
                if artifact_sensitivity["is_artifact_driven"]:
                    print(f"  ⚠ Labeled as artifact-driven (delta > 10%)")
            else:
                artifact_sensitivity = {"error": "Could not compute artifact sensitivity"}
        
        # Update tail_diagnosis with artifact sensitivity
        if "tail_diagnosis" in metrics:
            metrics["tail_diagnosis"]["artifact_sensitivity"] = artifact_sensitivity if artifact_sensitivity else None
        
        # Store results
        all_results[ref_model_name] = {
            **metrics,
            "artifact_metrics": artifact_metrics,
            "diversity_metrics": diversity_metrics,
            "entropy_metrics": entropy_metrics,
            "distributional_metrics": distributional_metrics if distributional_metrics else None,
            "byte_validity": byte_validity,
            "bootstrap_cis": bootstrap_cis,
            "worst_tokens": worst_tokens_dict,
            "per_sample_metrics": per_sample_metrics if cfg.get("save_per_sample_metrics", True) else None,
            "token_id_degeneracy": token_id_degeneracy,  # Only non-None for generated samples
        }
        
        # Store worst_tokens objects separately for report generation (not JSON-serializable, only for generated samples)
        if not is_real_text:
            worst_tokens_by_model[ref_model_name] = worst_tokens
        
        # Print summary
        print(f"\nResults for {ref_model_name}:")
        print(f"  Scoring mode: {metrics.get('scoring_mode', 'text_based')}")
        print(f"  PPL: {metrics['ppl']:.2f}")
        if 'robust_metrics' in metrics and 'median_ppl' in metrics['robust_metrics']:
            print(f"  Median PPL: {metrics['robust_metrics']['median_ppl']:.2f}")
        print(f"  Median NLL: {metrics['median_nll']:.4f}")
        print(f"  P99 NLL: {metrics['tail_stats']['percentiles'].get('p99', 'N/A')}")
        print(f"  Max NLL: {metrics['tail_stats'].get('max_nll', 'N/A')}")
        print(f"  Catastrophe rate (>15): {metrics['tail_stats']['catastrophe_rates'].get('catastrophe_rate_15', 0):.4f}")
        print(f"  Distinct-1 tokens: {diversity_metrics.get('distinct_1_tokens', 0):.4f}")
        print(f"  Avg sequence entropy: {entropy_metrics.get('avg_sequence_entropy', 0):.4f}")
        print(f"  Non-ASCII ratio: {artifact_metrics.get('non_ascii_ratio', 0):.4f}")
        if byte_validity:
            print(f"  Round-trip mismatch rate: {byte_validity.get('roundtrip_mismatch_rate', 0):.4f}")
        if token_id_degeneracy:
            # Support both single-dict and multi-n formats
            if "token_id_duplicate_rate" in token_id_degeneracy:
                # Legacy format (single n-gram order)
                print(f"  Token-ID duplicate rate: {token_id_degeneracy.get('token_id_duplicate_rate', 0):.4f}")
                print(f"  Repeated n-gram frac (mean): {token_id_degeneracy.get('rep_ngram_frac_mean', 0):.4f}")
                print(f"  Inter-sample Jaccard: {token_id_degeneracy.get('inter_sample_jaccard_mean', 0):.4f}")
            else:
                # New multi-n format: dict keyed by "n{n}" -> metrics
                # Use any entry to report duplicate rate (independent of n)
                any_metrics = next(iter(token_id_degeneracy.values()))
                dup_rate = any_metrics.get("token_id_duplicate_rate", 0.0)
                print(f"  Token-ID duplicate rate: {dup_rate:.4f}")
                # Then print per-n summary
                for key in sorted(token_id_degeneracy.keys()):
                    deg = token_id_degeneracy[key]
                    n = deg.get("rep_ngram_n", key)
                    print(f"  Repeated {n}-gram frac (mean): {deg.get('rep_ngram_frac_mean', 0):.4f}")
                    print(f"  Inter-sample Jaccard (n={n}): {deg.get('inter_sample_jaccard_mean', 0):.4f}")
        
        # Print tail diagnosis summary
        tail_diagnosis = metrics.get("tail_diagnosis", {})
        if tail_diagnosis:
            worst_tokens_list = tail_diagnosis.get("worst_tokens", [])
            if worst_tokens_list:
                print(f"\n  Tail Diagnosis Summary:")
                print(f"    Top-{len(worst_tokens_list)} worst tokens extracted")
                # Count artifact flags
                flag_counts = {
                    "has_unicode_replacement": sum(1 for wt in worst_tokens_list if wt.get("artifact_flags", {}).get("has_unicode_replacement", False)),
                    "non_ascii": sum(1 for wt in worst_tokens_list if wt.get("artifact_flags", {}).get("non_ascii", False)),
                    "weird_whitespace": sum(1 for wt in worst_tokens_list if wt.get("artifact_flags", {}).get("weird_whitespace", False)),
                    "long_alnum_run": sum(1 for wt in worst_tokens_list if wt.get("artifact_flags", {}).get("long_alnum_run", False)),
                    "broken_markup_hint": sum(1 for wt in worst_tokens_list if wt.get("artifact_flags", {}).get("broken_markup_hint", False)),
                }
                print(f"    Artifact flags in worst tokens:")
                for flag, count in flag_counts.items():
                    if count > 0:
                        print(f"      {flag}: {count}/{len(worst_tokens_list)}")
                
                # Show top 5 worst tokens
                print(f"    Top 5 worst tokens:")
                for wt in worst_tokens_list[:5]:
                    flags_str = ", ".join([k for k, v in wt.get("artifact_flags", {}).items() if v])
                    flags_str = flags_str if flags_str else "none"
                    print(f"      Rank {wt.get('rank', '?')}: NLL={wt.get('nll', 0):.2f}, token={wt.get('token_str', '?')}, flags=[{flags_str}]")
            
            artifact_sens = tail_diagnosis.get("artifact_sensitivity")
            if artifact_sens and artifact_sens.get("is_artifact_driven"):
                print(f"    ⚠ Artifact-driven PPL detected (delta: {artifact_sens.get('ppl_delta_pct', 0):.1f}%)")
    
    # Build final metrics dictionary
    final_metrics = {
        "file": Path(cfg.samples_path).stem if not is_real_text and hasattr(cfg, "samples_path") and cfg.samples_path else f"real_text_{cfg.real_text_dataset}",
        "model_tokenizer": cfg.model_tokenizer,
        "num_samples": len(texts),
        "is_real_text_baseline": is_real_text,
        "reference_models": reference_models,
        "results_by_model": all_results,
        "config": {
            "retokenize": bool(cfg.retokenize),
            "first_chunk_only": bool(cfg.first_chunk_only),
            "prompt_len": int(cfg.get("prompt_len", 0)) if cfg.get("prompt_len") is not None else None,
            "ignore_prefix_tokens": int(cfg.get("ignore_prefix_tokens", 0)),
            "max_length": int(cfg.max_length),
            "batch_size": int(cfg.batch_size),
        },
        "metadata": {
            "git_hash": _get_git_hash(),
            "hostname": socket.gethostname(),
            "timestamp": datetime.now().isoformat(),
            "python_version": sys.version,
        },
    }
    
    # Save metrics
    out_path = Path(hydra.utils.to_absolute_path(cfg.metrics_path))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(final_metrics, f, indent=2)
    print(f"\nSaved metrics to: {out_path}")
    
    # Save decoded samples
    if cfg.get("save_decoded_samples", False):
        samples_path = out_path.parent / f"{out_path.stem}_decoded_samples.txt"
        with open(samples_path, "w", encoding="utf-8") as f:
            for i, text in enumerate(texts):
                f.write(f"Sample {i}:\n{text}\n{'=' * 80}\n")
        print(f"Saved decoded samples to: {samples_path}")
    
    # Save worst tokens report (for each reference model)
    for ref_model_name, worst_tokens in worst_tokens_by_model.items():
        if worst_tokens and not is_real_text:  # Don't generate worst tokens report for real text baseline
            worst_tokens_path = out_path.parent / f"{out_path.stem}_worst_tokens_{ref_model_name.replace('/', '_').replace('-', '_')}.txt"
            with open(worst_tokens_path, "w", encoding="utf-8") as f:
                f.write(f"WORST TOKENS REPORT - {ref_model_name}\n")
                f.write("Top 50 highest NLL tokens\n")
                f.write("=" * 80 + "\n\n")
                f.write("This report shows the tokens with the highest NLL (negative log-likelihood)\n")
                f.write("under the reference model. These are the 'PPL-killer' tokens that drive\n")
                f.write("perplexity up. Check for:\n")
                f.write("- Unicode artifacts (replacement characters, control chars)\n")
                f.write("- Rare tokens or out-of-vocabulary tokens\n")
                f.write("- Tokenization mismatches\n")
                f.write("- Degenerate patterns\n")
                f.write("\n" + "=" * 80 + "\n\n")
                for i, wt in enumerate(worst_tokens[:50], 1):
                    f.write(f"Rank {i}:\n")
                    f.write(f"  Sample: {wt.sample_idx}, Token position: {wt.token_idx}\n")
                    f.write(f"  Token ID: {wt.token_id}, Token: {repr(wt.token_str)}\n")
                    f.write(f"  NLL: {wt.nll:.4f}\n")
                    f.write(f"  Context: ...{wt.context_before}[{wt.token_str}]{wt.context_after}...\n")
                    f.write("\n")
            print(f"Saved worst tokens report to: {worst_tokens_path}")
    
    # Save per-sample metrics if requested (from first reference model)
    if cfg.get("save_per_sample_metrics", True) and all_results:
        first_model_results = list(all_results.values())[0]
        per_sample_metrics = first_model_results.get("per_sample_metrics")
        if per_sample_metrics:
            per_sample_path = out_path.parent / f"{out_path.stem}_per_sample_metrics.jsonl"
            with open(per_sample_path, "w") as f:
                for metrics in per_sample_metrics:
                    f.write(json.dumps(metrics) + "\n")
            print(f"Saved per-sample metrics to: {per_sample_path}")
    
    # Generate diagnostic report
    diagnostic_path = out_path.parent / f"{out_path.stem}_diagnostic_report.txt"
    with open(diagnostic_path, "w", encoding="utf-8") as f:
        f.write("=" * 80 + "\n")
        f.write("DIAGNOSTIC REPORT - Cross-PPL Evaluation\n")
        f.write("=" * 80 + "\n\n")
        
        for ref_model_name, result in all_results.items():
            f.write(f"\nReference Model: {ref_model_name}\n")
            f.write("-" * 80 + "\n\n")
            
            # Scoring mode
            scoring_mode = result.get("scoring_mode", "text_based")
            f.write(f"Scoring Mode: {scoring_mode}\n")
            if scoring_mode == "token_id_direct":
                f.write("  ✓ Direct token-ID scoring (no decode/retokenize artifacts)\n")
                text_comp = result.get("text_based_comparison", {})
                if text_comp:
                    f.write(f"  Text-based PPL (for comparison): {text_comp.get('ppl', 'N/A'):.2f}\n")
                    f.write(f"  PPL delta (text - token_id): {text_comp.get('ppl_delta', 0):.2f}\n")
            else:
                f.write("  ⚠ Text-based scoring (decode+retokenize may introduce artifacts)\n")
            
            f.write(f"\nCore Metrics:\n")
            f.write(f"  PPL: {result.get('ppl', 'N/A'):.2f}\n")
            f.write(f"  Median PPL: {result.get('robust_metrics', {}).get('median_ppl', 'N/A'):.2f}\n")
            f.write(f"  Avg NLL: {result.get('avg_nll', 'N/A'):.4f}\n")
            f.write(f"  Median NLL: {result.get('median_nll', 'N/A'):.4f}\n")
            
            # Tail analysis
            tail_stats = result.get("tail_stats", {})
            if tail_stats:
                f.write(f"\nTail Analysis:\n")
                percentiles = tail_stats.get("percentiles", {})
                f.write(f"  P99 NLL: {percentiles.get('p99', 'N/A'):.2f}\n")
                f.write(f"  Max NLL: {tail_stats.get('max_nll', 'N/A'):.2f}\n")
                catastrophe_rates = tail_stats.get("catastrophe_rates", {})
                f.write(f"  Catastrophe rate (>15): {catastrophe_rates.get('catastrophe_rate_15', 0):.4f}\n")
                
                tail_contrib = result.get("tail_contribution", {})
                if tail_contrib:
                    f.write(f"\nTail Contribution (fraction of total NLL):\n")
                    f.write(f"  Top 10 tokens: {tail_contrib.get('tail_contribution_top_10', 0):.2%}\n")
                    f.write(f"  Top 50 tokens: {tail_contrib.get('tail_contribution_top_50', 0):.2%}\n")
                    f.write(f"  Top 0.1% tokens: {tail_contrib.get('tail_contribution_top_0.1pct', 0):.2%}\n")
            
            # Robust metrics
            robust = result.get("robust_metrics", {})
            if robust:
                f.write(f"\nRobust Metrics (tail-insensitive):\n")
                trimmed = robust.get("trimmed_means", {})
                if trimmed:
                    f.write(f"  Trimmed mean (0.1%): {trimmed.get('trimmed_mean_nll_0.1pct', 'N/A'):.4f}\n")
                    f.write(f"  Trimmed mean (1%): {trimmed.get('trimmed_mean_nll_1.0pct', 'N/A'):.4f}\n")
                winsorized = robust.get("winsorized_means", {})
                if winsorized:
                    f.write(f"  Winsorized mean (0.1%): {winsorized.get('winsorized_mean_nll_0.1pct', 'N/A'):.4f}\n")
            
            # Token type stratified
            token_type = result.get("token_type_stats", {})
            if token_type:
                f.write(f"\nToken-Type Stratified NLL:\n")
                for cat in ["whitespace_leading", "punctuation", "alphanumeric", "non_ascii"]:
                    avg_nll = token_type.get(f"{cat}_avg_nll", None)
                    fraction = token_type.get(f"{cat}_fraction", 0)
                    if avg_nll is not None and not np.isnan(avg_nll):
                        f.write(f"  {cat}: avg_nll={avg_nll:.4f}, fraction={fraction:.2%}\n")
            
            # Byte validity
            byte_validity = result.get("byte_validity", {})
            if byte_validity:
                f.write(f"\nByte-Level Validity:\n")
                f.write(f"  Invalid UTF-8 rate: {byte_validity.get('invalid_utf8_rate', 0):.4f}\n")
                f.write(f"  Replacement char rate: {byte_validity.get('replacement_char_rate', 0):.4f}\n")
                f.write(f"  Round-trip mismatch rate: {byte_validity.get('roundtrip_mismatch_rate', 0):.4f}\n")
                if byte_validity.get("roundtrip_mismatch_rate", 0) > 0.01:
                    f.write(f"  ⚠ WARNING: High round-trip mismatch! Decode/retokenize is changing tokens.\n")
            
            # Artifacts
            artifacts = result.get("artifact_metrics", {})
            if artifacts:
                f.write(f"\nArtifact Metrics:\n")
                f.write(f"  Non-ASCII ratio: {artifacts.get('non_ascii_ratio', 0):.4f}\n")
                f.write(f"  Unicode replacement rate: {artifacts.get('unicode_replacement_rate', 0):.4f}\n")
                f.write(f"  Samples with double spaces: {artifacts.get('samples_with_double_spaces', 0)}\n")
                f.write(f"  Samples with space before punct: {artifacts.get('samples_with_space_before_punct', 0)}\n")
                f.write(f"  Avg tokens per word: {artifacts.get('avg_tokens_per_word', 0):.2f}\n")
            
            # Diversity and entropy
            diversity_metrics = result.get("diversity_metrics", {})
            entropy_metrics = result.get("entropy_metrics", {})
            if diversity_metrics or entropy_metrics:
                f.write(f"\nDiversity & Entropy Metrics:\n")
                if diversity_metrics:
                    f.write(f"  Distinct-1 tokens: {diversity_metrics.get('distinct_1_tokens', 0):.4f}\n")
                    f.write(f"  Distinct-2 tokens: {diversity_metrics.get('distinct_2_tokens', 0):.4f}\n")
                    f.write(f"  Distinct-3 tokens: {diversity_metrics.get('distinct_3_tokens', 0):.4f}\n")
                if entropy_metrics:
                    f.write(f"  Avg sequence entropy: {entropy_metrics.get('avg_sequence_entropy', 0):.4f}\n")
                    f.write(f"  Median sequence entropy: {entropy_metrics.get('median_sequence_entropy', 0):.4f}\n")
                    f.write(f"  Min/Max sequence entropy: {entropy_metrics.get('min_sequence_entropy', 0):.4f} / {entropy_metrics.get('max_sequence_entropy', 0):.4f}\n")
            
            # Boundary error NLL contributions
            boundary_errors = result.get("boundary_error_stats", {})
            if boundary_errors:
                f.write(f"\nBoundary Error NLL Contributions:\n")
                f.write(f"  Boundary error fraction: {boundary_errors.get('boundary_error_fraction', 0):.2%}\n")
                f.write(f"  Boundary error avg NLL: {boundary_errors.get('boundary_error_avg_nll', 0):.4f}\n")
                f.write(f"  Boundary error NLL contribution: {boundary_errors.get('boundary_error_total_nll_contribution', 0):.2%}\n")
            
            # Bootstrap CIs
            bootstrap = result.get("bootstrap_cis", {})
            if bootstrap:
                f.write(f"\nBootstrap Confidence Intervals (95%):\n")
                f.write(f"  Avg NLL CI: [{bootstrap.get('avg_nll_bootstrap_ci_lower', 'N/A'):.4f}, {bootstrap.get('avg_nll_bootstrap_ci_upper', 'N/A'):.4f}]\n")
            
            # Token-ID Degeneracy Diagnostics
            degeneracy = result.get("token_id_degeneracy")
            if degeneracy:
                f.write(f"\nToken-ID Degeneracy Diagnostics:\n")
                # Support both single-n and multi-n formats
                if "token_id_duplicate_rate" in degeneracy:
                    # Legacy: single n-gram order
                    f.write(f"  Unique sequence ratio: {degeneracy.get('token_id_unique_ratio', 0):.4f}\n")
                    f.write(f"  Duplicate rate: {degeneracy.get('token_id_duplicate_rate', 0):.4f}\n")
                    ngram_n = degeneracy.get('rep_ngram_n', 4)
                    f.write(f"  Repeated {ngram_n}-gram fraction:\n")
                    f.write(f"    Mean: {degeneracy.get('rep_ngram_frac_mean', 0):.4f}\n")
                    f.write(f"    Median: {degeneracy.get('rep_ngram_frac_median', 0):.4f}\n")
                    f.write(f"    P95: {degeneracy.get('rep_ngram_frac_p95', 0):.4f}\n")
                    f.write(f"    Max: {degeneracy.get('rep_ngram_frac_max', 0):.4f}\n")
                    pair_sample = degeneracy.get('pair_sample', 0)
                    f.write(f"  Inter-sample Jaccard mean ({pair_sample} pairs): {degeneracy.get('inter_sample_jaccard_mean', 0):.4f}\n")

                    # Flag potential issues
                    if degeneracy.get('token_id_duplicate_rate', 0) > 0.1:
                        f.write(f"  ⚠ WARNING: High duplicate rate (>{10}%) indicates potential mode collapse\n")
                    if degeneracy.get('rep_ngram_frac_p95', 0) > 0.5:
                        f.write(f"  ⚠ WARNING: High intra-sample repetition (P95>{50}%)\n")
                    if degeneracy.get('inter_sample_jaccard_mean', 0) > 0.3:
                        f.write(f"  ⚠ WARNING: High inter-sample similarity (Jaccard>{0.3}) indicates low diversity\n")
                else:
                    # New: multiple n-gram orders stored as {"n2": {...}, "n3": {...}, ...}
                    # Use any entry to report unique/duplicate (independent of n)
                    any_metrics = next(iter(degeneracy.values()))
                    f.write(f"  Unique sequence ratio: {any_metrics.get('token_id_unique_ratio', 0):.4f}\n")
                    f.write(f"  Duplicate rate: {any_metrics.get('token_id_duplicate_rate', 0):.4f}\n")

                    # Then, for each n, print repeated n-gram stats and Jaccard
                    for key in sorted(degeneracy.keys()):
                        deg = degeneracy[key]
                        n = deg.get('rep_ngram_n', key)
                        f.write(f"  Repeated {n}-gram fraction:\n")
                        f.write(f"    Mean: {deg.get('rep_ngram_frac_mean', 0):.4f}\n")
                        f.write(f"    Median: {deg.get('rep_ngram_frac_median', 0):.4f}\n")
                        f.write(f"    P95: {deg.get('rep_ngram_frac_p95', 0):.4f}\n")
                        f.write(f"    Max: {deg.get('rep_ngram_frac_max', 0):.4f}\n")
                        pair_sample = deg.get('pair_sample', 0)
                        f.write(f"  Inter-sample Jaccard mean ({pair_sample} pairs, n={n}): {deg.get('inter_sample_jaccard_mean', 0):.4f}\n")

                    # Flag potential issues using the highest-order n-gram stats (most stringent)
                    last_key = sorted(degeneracy.keys())[-1]
                    high_deg = degeneracy[last_key]
                    if high_deg.get('token_id_duplicate_rate', 0) > 0.1:
                        f.write(f"  ⚠ WARNING: High duplicate rate (>{10}%) indicates potential mode collapse\n")
                    if high_deg.get('rep_ngram_frac_p95', 0) > 0.5:
                        f.write(f"  ⚠ WARNING: High intra-sample repetition (P95>{50}%)\n")
                    if high_deg.get('inter_sample_jaccard_mean', 0) > 0.3:
                        f.write(f"  ⚠ WARNING: High inter-sample similarity (Jaccard>{0.3}) indicates low diversity\n")
    
    print(f"Saved diagnostic report to: {diagnostic_path}")
    
    return final_metrics


if __name__ == "__main__":
    main()
