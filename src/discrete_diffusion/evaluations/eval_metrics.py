"""Evaluation metrics utilities for generative perplexity evaluation.

This module provides comprehensive metrics for analyzing generated text:
- Token-level NLL tail analysis
- Artifact detection (Unicode, whitespace, BPE fragmentation)
- Diversity and degeneracy metrics
- Distributional similarity (optional)
"""

from __future__ import annotations

import re
import unicodedata
from collections import Counter
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from transformers import AutoTokenizer


@dataclass
class WorstTokenInfo:
    """Information about a high-NLL token."""
    sample_idx: int
    token_idx: int
    token_id: int
    token_str: str
    nll: float
    context_before: str
    context_after: str


def compute_nll_tail_stats(
    all_nlls: List[float],
    thresholds: List[float] = [10.0, 15.0, 20.0],
    top_k_values: List[int] = [10, 50, 100],
) -> Dict[str, Any]:
    """Compute comprehensive NLL tail statistics."""
    if not all_nlls:
        return {}
    
    sorted_nlls = sorted(all_nlls)
    n = len(sorted_nlls)
    
    # Extended percentiles
    percentiles = {}
    for p in [10, 25, 50, 75, 90, 95, 99]:
        idx = int(p / 100.0 * n) if n > 0 else 0
        percentiles[f"p{p}"] = sorted_nlls[idx] if idx < n else float("nan")
    
    # Max NLL
    max_nll = float(sorted_nlls[-1]) if sorted_nlls else float("nan")
    
    # Top-K means
    top_k_means = {}
    for k in top_k_values:
        if n >= k:
            top_k_means[f"top_{k}_mean"] = float(np.mean(sorted_nlls[-k:]))
        else:
            top_k_means[f"top_{k}_mean"] = float(np.mean(sorted_nlls)) if sorted_nlls else float("nan")
    
    # Catastrophe rates
    catastrophe_rates = {}
    for threshold in thresholds:
        count = sum(1 for nll in all_nlls if nll > threshold)
        catastrophe_rates[f"catastrophe_rate_{int(threshold)}"] = count / n if n > 0 else 0.0
    
    return {
        "percentiles": percentiles,
        "max_nll": max_nll,
        "top_k_means": top_k_means,
        "catastrophe_rates": catastrophe_rates,
        "total_tokens": n,
    }


def extract_worst_tokens(
    nll_tensor: torch.Tensor,
    input_ids: torch.Tensor,
    valid_mask: torch.Tensor,
    texts: List[str],
    tokenizer: AutoTokenizer,
    top_k: int = 50,
    context_chars: int = 40,
    batch_offset: int = 0,
) -> List[WorstTokenInfo]:
    """Extract worst (highest NLL) tokens with context.
    
    Args:
        nll_tensor: [B, L-1] NLL values
        input_ids: [B, L] input token IDs
        valid_mask: [B, L-1] boolean mask for valid tokens
        texts: List of decoded text strings (length B)
        tokenizer: Tokenizer for decoding
        top_k: Number of worst tokens to extract
        context_chars: Number of characters for context window
        batch_offset: Offset to add to batch_idx for global sample indexing
    """
    # Flatten and get indices
    valid_nlls = nll_tensor[valid_mask.bool()]
    if len(valid_nlls) == 0:
        return []
    
    valid_indices = torch.nonzero(valid_mask.bool(), as_tuple=False)
    
    # Get top-K worst tokens
    k = min(top_k, len(valid_nlls))
    top_k_indices = torch.topk(valid_nlls, k=k).indices
    
    worst_tokens = []
    for idx in top_k_indices:
        batch_idx_local, token_pos = valid_indices[idx.item()].cpu().tolist()
        batch_idx_global = batch_offset + batch_idx_local
        # token_pos is in labels (shifted), so the actual token is at token_pos + 1 in input_ids
        token_id = input_ids[batch_idx_local, token_pos + 1].item()
        nll_val = valid_nlls[idx].item()
        
        # Get token string
        token_str = tokenizer.decode([token_id], skip_special_tokens=False, clean_up_tokenization_spaces=False)
        
        # Get context from decoded text
        if batch_idx_local < len(texts):
            text = texts[batch_idx_local]
            # Decode tokens up to (but not including) the token of interest
            tokens_before = input_ids[batch_idx_local, :token_pos + 1].tolist()
            decoded_before = tokenizer.decode(tokens_before, skip_special_tokens=True, clean_up_tokenization_spaces=False)
            token_start = len(decoded_before)
            
            # Decode including the token to find its end
            tokens_up_to = input_ids[batch_idx_local, :token_pos + 2].tolist()
            decoded_up_to = tokenizer.decode(tokens_up_to, skip_special_tokens=True, clean_up_tokenization_spaces=False)
            token_end = len(decoded_up_to)
            
            # Extract context
            context_before = text[max(0, token_start - context_chars):token_start]
            context_after = text[token_end:min(len(text), token_end + context_chars)]
        else:
            context_before = ""
            context_after = ""
        
        worst_tokens.append(WorstTokenInfo(
            sample_idx=batch_idx_global,
            token_idx=token_pos,
            token_id=token_id,
            token_str=token_str,
            nll=nll_val,
            context_before=context_before,
            context_after=context_after,
        ))
    
    return worst_tokens


def compute_artifact_metrics(
    texts: List[str],
    tokenizer: AutoTokenizer,
) -> Dict[str, Any]:
    """Compute artifact and text hygiene metrics."""
    unicode_replacement_count = 0
    control_char_count = 0
    zero_width_count = 0
    non_ascii_chars = 0
    total_chars = 0
    
    double_space_count = 0
    space_before_punct_count = 0
    missing_space_after_punct_count = 0
    newline_anomaly_count = 0
    
    special_token_strings = set()
    if hasattr(tokenizer, "special_tokens_map"):
        special_token_strings.update(tokenizer.special_tokens_map.values())
    if hasattr(tokenizer, "added_tokens_decoder"):
        for token_id, token_info in tokenizer.added_tokens_decoder.items():
            if hasattr(token_info, "content"):
                special_token_strings.add(token_info.content)
    
    samples_with_special_strings = 0
    
    # BPE fragmentation proxies
    tokens_per_word_list = []
    tokens_per_char_list = []
    
    for text in texts:
        # Unicode analysis
        for char in text:
            total_chars += 1
            if char == "\ufffd":  # Unicode replacement character
                unicode_replacement_count += 1
            if unicodedata.category(char) == "Cc":  # Control characters
                control_char_count += 1
            if char in ["\u200b", "\u200c", "\u200d", "\ufeff"]:  # Zero-width spaces
                zero_width_count += 1
            if ord(char) > 127:  # Non-ASCII
                non_ascii_chars += 1
        
        # Whitespace anomalies
        if "  " in text:  # Double spaces
            double_space_count += 1
        if re.search(r"\s+[,\.!?;:]", text):  # Space before punctuation
            space_before_punct_count += 1
        if re.search(r"[a-zA-Z][,\.!?;:][a-zA-Z]", text):  # Missing space after punctuation
            missing_space_after_punct_count += 1
        if "\n\n\n" in text or text.count("\n") > len(text) / 50:  # Newline anomalies
            newline_anomaly_count += 1
        
        # Special token strings
        has_special = False
        for special_str in special_token_strings:
            if special_str and special_str in text:
                has_special = True
                break
        if has_special:
            samples_with_special_strings += 1
        
        # BPE fragmentation
        words = text.split()
        tokens = tokenizer.encode(text, add_special_tokens=False)
        if len(words) > 0:
            tokens_per_word_list.append(len(tokens) / len(words))
        if len(text) > 0:
            tokens_per_char_list.append(len(tokens) / len(text))
    
    return {
        "unicode_replacement_count": unicode_replacement_count,
        "unicode_replacement_rate": unicode_replacement_count / total_chars if total_chars > 0 else 0.0,
        "control_char_count": control_char_count,
        "control_char_rate": control_char_count / total_chars if total_chars > 0 else 0.0,
        "zero_width_char_count": zero_width_count,
        "zero_width_char_rate": zero_width_count / total_chars if total_chars > 0 else 0.0,
        "non_ascii_chars": non_ascii_chars,
        "non_ascii_ratio": non_ascii_chars / total_chars if total_chars > 0 else 0.0,
        "samples_with_double_spaces": double_space_count,
        "samples_with_space_before_punct": space_before_punct_count,
        "samples_with_missing_space_after_punct": missing_space_after_punct_count,
        "samples_with_newline_anomalies": newline_anomaly_count,
        "samples_with_special_strings": samples_with_special_strings,
        "fraction_special_strings": samples_with_special_strings / len(texts) if texts else 0.0,
        "avg_tokens_per_word": float(np.mean(tokens_per_word_list)) if tokens_per_word_list else 0.0,
        "median_tokens_per_word": float(np.median(tokens_per_word_list)) if tokens_per_word_list else 0.0,
        "avg_tokens_per_char": float(np.mean(tokens_per_char_list)) if tokens_per_char_list else 0.0,
        "median_tokens_per_char": float(np.median(tokens_per_char_list)) if tokens_per_char_list else 0.0,
    }


def compute_enhanced_diversity_metrics(
    texts: List[str],
    tokenizer: AutoTokenizer,
) -> Dict[str, Any]:
    """Compute enhanced diversity and degeneracy metrics."""
    all_token_ngrams_1 = []
    all_token_ngrams_2 = []
    all_token_ngrams_3 = []
    all_word_ngrams_1 = []
    all_word_ngrams_2 = []
    all_word_ngrams_3 = []
    
    # Per-sample metrics
    unique_token_masses = []
    top_token_dominances = []
    repetition_rates_2 = []
    repetition_rates_3 = []
    repetition_rates_4 = []
    max_repeated_ngram_counts = []
    
    for text in texts:
        tokens = tokenizer.encode(text, add_special_tokens=False)
        words = text.split()
        
        # Token-level ngrams
        if len(tokens) > 0:
            all_token_ngrams_1.extend(tokens)
        if len(tokens) > 1:
            all_token_ngrams_2.extend(zip(tokens[:-1], tokens[1:]))
        if len(tokens) > 2:
            all_token_ngrams_3.extend(zip(tokens[:-2], tokens[1:-1], tokens[2:]))
        
        # Word-level ngrams
        if len(words) > 0:
            all_word_ngrams_1.extend(words)
        if len(words) > 1:
            all_word_ngrams_2.extend(zip(words[:-1], words[1:]))
        if len(words) > 2:
            all_word_ngrams_3.extend(zip(words[:-2], words[1:-1], words[2:]))
        
        # Unique token mass
        if len(tokens) > 0:
            token_counts = Counter(tokens)
            unique_count = sum(1 for count in token_counts.values() if count == 1)
            unique_token_masses.append(unique_count / len(tokens))
        else:
            unique_token_masses.append(0.0)
        
        # Top token dominance
        if len(tokens) > 0:
            token_counts = Counter(tokens)
            max_count = max(token_counts.values())
            top_token_dominances.append(max_count / len(tokens))
        else:
            top_token_dominances.append(0.0)
        
        # Repetition rates (2-gram, 3-gram, 4-gram)
        for n in [2, 3, 4]:
            if len(tokens) >= n:
                ngrams = [tuple(tokens[i:i + n]) for i in range(len(tokens) - n + 1)]
                ngram_counts = Counter(ngrams)
                repeated = sum(1 for count in ngram_counts.values() if count > 1)
                rate = repeated / len(ngram_counts) if ngram_counts else 0.0
                if n == 2:
                    repetition_rates_2.append(rate)
                elif n == 3:
                    repetition_rates_3.append(rate)
                else:
                    repetition_rates_4.append(rate)
                max_repeated_ngram_counts.append(max(ngram_counts.values()) if ngram_counts else 0)
            else:
                if n == 2:
                    repetition_rates_2.append(0.0)
                elif n == 3:
                    repetition_rates_3.append(0.0)
                else:
                    repetition_rates_4.append(0.0)
                max_repeated_ngram_counts.append(0)
    
    def distinct(ngrams: List) -> float:
        if not ngrams:
            return 0.0
        unique = len(set(ngrams))
        total = len(ngrams)
        return unique / total if total > 0 else 0.0
    
    return {
        "distinct_1_tokens": distinct(all_token_ngrams_1),
        "distinct_2_tokens": distinct(all_token_ngrams_2),
        "distinct_3_tokens": distinct(all_token_ngrams_3),
        "distinct_1_words": distinct(all_word_ngrams_1),
        "distinct_2_words": distinct(all_word_ngrams_2),
        "distinct_3_words": distinct(all_word_ngrams_3),
        "avg_unique_token_mass": float(np.mean(unique_token_masses)) if unique_token_masses else 0.0,
        "median_unique_token_mass": float(np.median(unique_token_masses)) if unique_token_masses else 0.0,
        "avg_top_token_dominance": float(np.mean(top_token_dominances)) if top_token_dominances else 0.0,
        "median_top_token_dominance": float(np.median(top_token_dominances)) if top_token_dominances else 0.0,
        "repetition_rate_2gram": float(np.mean(repetition_rates_2)) if repetition_rates_2 else 0.0,
        "repetition_rate_3gram": float(np.mean(repetition_rates_3)) if repetition_rates_3 else 0.0,
        "repetition_rate_4gram": float(np.mean(repetition_rates_4)) if repetition_rates_4 else 0.0,
        "max_repeated_ngram_count": int(np.max(max_repeated_ngram_counts)) if max_repeated_ngram_counts else 0,
        "avg_max_repeated_ngram_count": float(np.mean(max_repeated_ngram_counts)) if max_repeated_ngram_counts else 0.0,
    }


def compute_sequence_entropy(tokens: List[int]) -> float:
    """Compute entropy of token distribution in a sequence.
    
    Entropy measures the diversity/unpredictability of the token distribution.
    Higher entropy indicates more diverse token usage.
    
    Args:
        tokens: List of token IDs
        
    Returns:
        Entropy value (bits)
    """
    if len(tokens) == 0:
        return 0.0
    token_counts = Counter(tokens)
    probs = np.array([count / len(tokens) for count in token_counts.values()])
    # Entropy: -sum(p * log(p))
    entropy = -np.sum(probs * np.log(probs + 1e-10))
    return float(entropy)


def compute_entropy_metrics(
    texts: List[str],
    tokenizer: AutoTokenizer,
) -> Dict[str, Any]:
    """Compute entropy metrics for generated samples.
    
    Computes sequence-level entropy for each sample, measuring the diversity
    of token usage within each sequence.
    
    Args:
        texts: List of text samples
        tokenizer: Tokenizer to encode texts
        
    Returns:
        Dictionary with entropy statistics
    """
    sequence_entropies = []
    
    for text in texts:
        tokens = tokenizer.encode(text, add_special_tokens=False)
        if len(tokens) > 0:
            entropy = compute_sequence_entropy(tokens)
            sequence_entropies.append(entropy)
    
    if not sequence_entropies:
        return {
            "avg_sequence_entropy": 0.0,
            "median_sequence_entropy": 0.0,
            "min_sequence_entropy": 0.0,
            "max_sequence_entropy": 0.0,
            "std_sequence_entropy": 0.0,
        }
    
    return {
        "avg_sequence_entropy": float(np.mean(sequence_entropies)),
        "median_sequence_entropy": float(np.median(sequence_entropies)),
        "min_sequence_entropy": float(np.min(sequence_entropies)),
        "max_sequence_entropy": float(np.max(sequence_entropies)),
        "std_sequence_entropy": float(np.std(sequence_entropies)),
    }


def compute_longest_repeated_substring(text: str) -> int:
    """Compute approximate longest repeated substring length (word-level)."""
    words = text.split()
    if len(words) < 2:
        return 0
    
    max_len = 0
    for i in range(len(words)):
        for j in range(i + 1, len(words)):
            k = 0
            while i + k < len(words) and j + k < len(words) and words[i + k] == words[j + k]:
                k += 1
            max_len = max(max_len, k)
    return max_len


def compute_embedding_distance(
    texts: List[str],
    reference_texts: Optional[List[str]] = None,
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
) -> Optional[Dict[str, float]]:
    """Compute embedding-based distance metrics (optional dependency)."""
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError:
        return None
    
    try:
        model = SentenceTransformer(model_name)
        embeddings = model.encode(texts, show_progress_bar=False, convert_to_numpy=True)
        
        if reference_texts is not None:
            ref_embeddings = model.encode(reference_texts, show_progress_bar=False, convert_to_numpy=True)
            # Compute cosine distances
            from sklearn.metrics.pairwise import cosine_similarity
            similarities = cosine_similarity(embeddings, ref_embeddings)
            mean_similarity = float(np.mean(similarities))
            mean_distance = 1.0 - mean_similarity
            return {
                "embedding_mean_cosine_similarity": mean_similarity,
                "embedding_mean_cosine_distance": mean_distance,
            }
        else:
            # Just return embedding stats
            return {
                "embedding_computed": True,
                "embedding_dim": embeddings.shape[1],
            }
    except Exception as e:
        return {"embedding_error": str(e)}


def compute_mauve_score(
    generated_texts: List[str],
    reference_texts: List[str],
) -> Optional[Dict[str, float]]:
    """Compute MAUVE score (optional dependency)."""
    try:
        import mauve
    except ImportError:
        return None
    
    try:
        out = mauve.compute_mauve(
            p_text=generated_texts,
            q_text=reference_texts,
            device_id=0 if torch.cuda.is_available() else -1,
            verbose=False,
        )
        return {
            "mauve_score": float(out.mauve),
            "mauve_p_hist": out.p_hist.tolist() if hasattr(out, "p_hist") else None,
            "mauve_q_hist": out.q_hist.tolist() if hasattr(out, "q_hist") else None,
        }
    except Exception as e:
        return {"mauve_error": str(e)}


def check_tokenizer_compatibility(
    tokenizer1: AutoTokenizer,
    tokenizer2: AutoTokenizer,
) -> Tuple[bool, str]:
    """Check if two tokenizers are compatible for direct token-ID scoring.
    
    Returns:
        (is_compatible, reason)
    """
    # Check vocab size
    if tokenizer1.vocab_size != tokenizer2.vocab_size:
        return False, f"Vocab size mismatch: {tokenizer1.vocab_size} vs {tokenizer2.vocab_size}"
    
    # Check if vocabularies match (compare get_vocab() dictionaries)
    try:
        vocab1 = tokenizer1.get_vocab()
        vocab2 = tokenizer2.get_vocab()
        
        if vocab1 != vocab2:
            # Check if at least the token-to-id mapping is identical
            # (id-to-token might differ due to added tokens, but token-to-id should match)
            mismatches = 0
            for token, id1 in vocab1.items():
                if token in vocab2:
                    if vocab2[token] != id1:
                        mismatches += 1
                        if mismatches > 10:  # Don't check all, just sample
                            return False, f"Token-to-ID mapping mismatch (found {mismatches}+ mismatches)"
                else:
                    return False, f"Token '{token}' missing in second tokenizer"
            
            # If we got here, vocabularies are compatible
            return True, "Vocabularies match"
        else:
            return True, "Vocabularies identical"
    except Exception as e:
        # Fallback: check if they're the same type and have same vocab size
        if type(tokenizer1).__name__ == type(tokenizer2).__name__:
            return True, f"Same tokenizer type with matching vocab size ({tokenizer1.vocab_size})"
        return False, f"Cannot verify compatibility: {str(e)}"


def compute_robust_metrics(
    all_nlls: List[float],
    trim_levels: List[float] = [0.001, 0.005, 0.01],
) -> Dict[str, Any]:
    """Compute robust/tail-insensitive metrics."""
    if not all_nlls:
        return {}
    
    sorted_nlls = np.array(sorted(all_nlls))
    n = len(sorted_nlls)
    
    # Trimmed means
    trimmed_means = {}
    for trim_level in trim_levels:
        trim_count = int(n * trim_level)
        if trim_count > 0 and n > 2 * trim_count:
            trimmed = sorted_nlls[trim_count:-trim_count]
            trimmed_means[f"trimmed_mean_nll_{trim_level*100:.1f}pct"] = float(np.mean(trimmed))
        else:
            trimmed_means[f"trimmed_mean_nll_{trim_level*100:.1f}pct"] = float(np.mean(sorted_nlls))
    
    # Winsorized means (replace extreme values with trimmed values)
    winsorized_means = {}
    for trim_level in trim_levels:
        trim_count = int(n * trim_level)
        if trim_count > 0 and n > 2 * trim_count:
            winsorized = sorted_nlls.copy()
            winsorized[:trim_count] = sorted_nlls[trim_count]
            winsorized[-trim_count:] = sorted_nlls[-trim_count-1]
            winsorized_means[f"winsorized_mean_nll_{trim_level*100:.1f}pct"] = float(np.mean(winsorized))
        else:
            winsorized_means[f"winsorized_mean_nll_{trim_level*100:.1f}pct"] = float(np.mean(sorted_nlls))
    
    # Median PPL
    median_nll = float(np.median(sorted_nlls))
    median_ppl = float(np.exp(median_nll))
    
    return {
        "trimmed_means": trimmed_means,
        "winsorized_means": winsorized_means,
        "median_ppl": median_ppl,
    }


def compute_tail_contribution(
    all_nlls: List[float],
    top_k_values: List[int] = [10, 50, 100],
    top_percentiles: List[float] = [0.001, 0.01],
) -> Dict[str, float]:
    """Compute fraction of total NLL contributed by tail tokens."""
    if not all_nlls:
        return {}
    
    sorted_nlls = np.array(sorted(all_nlls))
    total_nll = float(np.sum(sorted_nlls))
    n = len(sorted_nlls)
    
    contributions = {}
    
    # Top-K contributions
    for k in top_k_values:
        if n >= k:
            top_k_nll = float(np.sum(sorted_nlls[-k:]))
            contributions[f"tail_contribution_top_{k}"] = top_k_nll / total_nll if total_nll > 0 else 0.0
    
    # Top percentile contributions
    for pct in top_percentiles:
        count = max(1, int(n * pct))
        top_pct_nll = float(np.sum(sorted_nlls[-count:]))
        contributions[f"tail_contribution_top_{pct*100:.1f}pct"] = top_pct_nll / total_nll if total_nll > 0 else 0.0
    
    return contributions


def compute_token_type_stratified_nll(
    token_ids: torch.Tensor,
    nll_tensor: torch.Tensor,
    valid_mask: torch.Tensor,
    tokenizer: AutoTokenizer,
) -> Dict[str, Any]:
    """Compute NLL stratified by token type categories."""
    valid_nlls = nll_tensor[valid_mask.bool()].detach().cpu().numpy()
    valid_token_ids = token_ids[valid_mask.bool()].detach().cpu().numpy()
    
    if len(valid_nlls) == 0:
        return {}
    
    # Categorize tokens
    categories = {
        "whitespace_leading": [],
        "punctuation": [],
        "alphanumeric": [],
        "non_ascii": [],
        "control_like": [],
        "other": [],
    }
    
    for token_id, nll_val in zip(valid_token_ids, valid_nlls):
        try:
            token_str = tokenizer.decode([int(token_id)], skip_special_tokens=False, clean_up_tokenization_spaces=False)
            
            # Categorize
            if token_str.strip() == "" or token_str.startswith(" ") or token_str.startswith("\n"):
                categories["whitespace_leading"].append(float(nll_val))
            elif any(c in token_str for c in ",.!?;:()[]{}\"'"):
                categories["punctuation"].append(float(nll_val))
            elif token_str.isalnum() or (token_str.replace(" ", "").isalnum()):
                categories["alphanumeric"].append(float(nll_val))
            elif any(ord(c) > 127 for c in token_str):
                categories["non_ascii"].append(float(nll_val))
            elif any(ord(c) < 32 and c not in "\n\t\r" for c in token_str):
                categories["control_like"].append(float(nll_val))
            else:
                categories["other"].append(float(nll_val))
        except:
            categories["other"].append(float(nll_val))
    
    # Compute stats per category
    category_stats = {}
    total_tokens = len(valid_nlls)
    
    for cat_name, nlls in categories.items():
        if nlls:
            category_stats[f"{cat_name}_avg_nll"] = float(np.mean(nlls))
            category_stats[f"{cat_name}_median_nll"] = float(np.median(nlls))
            category_stats[f"{cat_name}_count"] = len(nlls)
            category_stats[f"{cat_name}_fraction"] = len(nlls) / total_tokens if total_tokens > 0 else 0.0
        else:
            category_stats[f"{cat_name}_avg_nll"] = float("nan")
            category_stats[f"{cat_name}_median_nll"] = float("nan")
            category_stats[f"{cat_name}_count"] = 0
            category_stats[f"{cat_name}_fraction"] = 0.0
    
    return category_stats


def compute_byte_level_validity(
    texts: List[str],
    token_ids_original: np.ndarray,
    tokenizer: AutoTokenizer,
    num_check: int = 100,
) -> Dict[str, Any]:
    """Compute byte-level validity diagnostics for GPT2 byte-BPE."""
    invalid_utf8_count = 0
    replacement_char_count = 0
    roundtrip_mismatches = 0
    roundtrip_mismatch_examples = []
    
    check_count = min(num_check, len(texts))
    
    for i in range(check_count):
        text = texts[i]
        original_tokens = token_ids_original[i] if i < len(token_ids_original) else None
        
        # Check for invalid UTF-8
        try:
            text.encode('utf-8').decode('utf-8')
        except UnicodeDecodeError:
            invalid_utf8_count += 1
        
        # Check for replacement characters
        if "\ufffd" in text:
            replacement_char_count += 1
        
        # Check round-trip
        if original_tokens is not None:
            try:
                # Decode original tokens
                decoded = tokenizer.decode(original_tokens.tolist(), skip_special_tokens=False, clean_up_tokenization_spaces=False)
                # Re-encode
                reencoded = tokenizer.encode(decoded, add_special_tokens=False)
                # Compare
                original_list = original_tokens.tolist()
                if len(reencoded) != len(original_list) or any(a != b for a, b in zip(reencoded, original_list)):
                    roundtrip_mismatches += 1
                    if len(roundtrip_mismatch_examples) < 5:
                        roundtrip_mismatch_examples.append({
                            "sample_idx": i,
                            "original_len": len(original_list),
                            "reencoded_len": len(reencoded),
                            "first_mismatch": next(
                                (j for j, (a, b) in enumerate(zip(original_list[:20], reencoded[:20])) if a != b),
                                None
                            ),
                        })
            except Exception:
                roundtrip_mismatches += 1
    
    return {
        "invalid_utf8_count": invalid_utf8_count,
        "invalid_utf8_rate": invalid_utf8_count / check_count if check_count > 0 else 0.0,
        "replacement_char_count": replacement_char_count,
        "replacement_char_rate": replacement_char_count / check_count if check_count > 0 else 0.0,
        "roundtrip_mismatch_count": roundtrip_mismatches,
        "roundtrip_mismatch_rate": roundtrip_mismatches / check_count if check_count > 0 else 0.0,
        "roundtrip_mismatch_examples": roundtrip_mismatch_examples[:5],
    }


def compute_bootstrap_ci(
    per_sample_metrics: List[Dict[str, Any]],
    metric_key: str = "avg_nll",
    n_bootstrap: int = 1000,
    confidence: float = 0.95,
    seed: int = 42,
) -> Dict[str, float]:
    """Compute bootstrap confidence intervals for a metric."""
    if not per_sample_metrics or metric_key not in per_sample_metrics[0]:
        return {}
    
    values = [m[metric_key] for m in per_sample_metrics if metric_key in m]
    if not values:
        return {}
    
    np.random.seed(seed)
    bootstrap_samples = []
    
    for _ in range(n_bootstrap):
        resampled = np.random.choice(values, size=len(values), replace=True)
        bootstrap_samples.append(float(np.mean(resampled)))
    
    bootstrap_samples = np.array(bootstrap_samples)
    alpha = 1 - confidence
    lower = float(np.percentile(bootstrap_samples, 100 * alpha / 2))
    upper = float(np.percentile(bootstrap_samples, 100 * (1 - alpha / 2)))
    
    return {
        f"{metric_key}_bootstrap_ci_lower": lower,
        f"{metric_key}_bootstrap_ci_upper": upper,
        f"{metric_key}_bootstrap_ci_level": confidence,
    }
