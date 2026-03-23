#!/usr/bin/env python3
"""Analyze SIK Kernel neighbors and transition probabilities.

Usage:
    python analyze_kernel.py --word "cat" --variable_bandwidth True --k_neighbors 7
"""

import argparse
import os
import sys
import torch
from pathlib import Path

# CRITICAL: Set environment variables BEFORE importing transformers
work_dir = os.environ.get("WORK", os.path.expanduser("~"))
hf_home = os.path.join(work_dir, ".cache", "huggingface")
os.environ["HF_HOME"] = hf_home
os.environ["HF_DATASETS_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_HUB_OFFLINE"] = "1"

from omegaconf import OmegaConf
from transformers import GPT2Model

# Add project root to path
repo_root = Path(__file__).resolve().parents[3]
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from src.discrete_diffusion.forward_process import SIKForwardProcess, KNNKernel, KeOpsKernel
from src.discrete_diffusion.noise_schedules import LogLinear
from src.discrete_diffusion.data import get_tokenizer


def load_gpt2_embeddings(tokenizer):
    try:
        model = GPT2Model.from_pretrained("gpt2", local_files_only=True)
    except (OSError, AttributeError) as exc:
        raise RuntimeError(
            "GPT-2 model weights were not found in the local Hugging Face cache. "
            "These visualization scripts run in offline mode, so cache GPT-2 first."
        ) from exc

    model.resize_token_embeddings(len(tokenizer))
    return model.wte.weight.detach()

def main():
    parser = argparse.ArgumentParser(description="Analyze SIK Kernel neighbors.")
    parser.add_argument("--word", action="append", help="Word(s) to analyze")
    parser.add_argument("--top_k", type=int, default=10, help="Number of neighbors to show")
    parser.add_argument("--metric", type=str, default="gaussian", choices=["gaussian", "cosine"])
    parser.add_argument("--implementation", type=str, default="knn", choices=["knn", "keops"])
    parser.add_argument("--epsilon", type=float, default=0.01)
    parser.add_argument("--beta", type=float, default=0.0, help="Temperature beta")
    parser.add_argument("--gamma", type=float, default=0.0)
    parser.add_argument("--variable_bandwidth", type=str, default="True", choices=["True", "False"])
    parser.add_argument("--k_neighbors", type=int, default=7, help="Neighbors for local sigma")
    args = parser.parse_args()

    vb = args.variable_bandwidth == "True"
    words = args.word if args.word else ["cat", "mat", "The", "running"]
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"Loading GPT-2 and tokenizer...")
    tokenizer = get_tokenizer(OmegaConf.create({"data": {"tokenizer_name_or_path": "gpt2"}}))
    embeddings = load_gpt2_embeddings(tokenizer).to(device)

    print(f"Building SIK Kernel ({args.metric}, implementation={args.implementation}, self_tuning={vb})...")
    kernel_params = {
        "embeddings": embeddings,
        "epsilon": args.epsilon,
        "gamma": args.gamma,
        "metric": args.metric,
        "variable_bandwidth": vb,
        "k_neighbors": args.k_neighbors,
    }
    
    if args.implementation == "knn":
        kernel = KNNKernel(**kernel_params, top_k=64)
    else:
        kernel = KeOpsKernel(**kernel_params)
        
    schedule = LogLinear(eps=1e-3)

    print("\n" + "="*110)
    print(f"SIK KERNEL ANALYSIS (eps={args.epsilon}, beta={args.beta}, self_tuning={vb})")
    print("="*110)

    for word in words:
        token_ids = tokenizer.encode(word, add_special_tokens=False)
        if not token_ids:
            token_ids = tokenizer.encode(" " + word, add_special_tokens=False)
        
        if not token_ids:
            print(f"\nCould not find token for '{word}'")
            continue
            
        tid = token_ids[0]
        actual_token = tokenizer.decode([tid])
        
        # Get neighbors and scores
        if args.implementation == "knn":
            neighbors = kernel._knn_indices[tid].cpu()
            logR = kernel._logR_vocab[tid].float().cpu()
        else:
            # For KeOps, calculate logR for visualization
            src_emb = embeddings[tid].unsqueeze(0)
            if args.metric == "gaussian":
                dist = torch.cdist(src_emb, embeddings, p=2).pow(2)
            else:
                dist = 1.0 - torch.mm(src_emb, embeddings.t())
            
            if vb:
                bw = kernel._sigma[tid] * kernel._sigma
                logK = -dist / (args.epsilon * bw)
            else:
                logK = -dist / args.epsilon
            n_i = kernel.n_i[tid]
            n_j = kernel.n_i
            logR = (logK - (n_i.log() + n_j.log())).squeeze(0).cpu()
            logR[tid] = float("-inf")
            logR_vals, neighbors = logR.topk(64)
            logR = logR_vals

        # Calculate raw distances for report
        tgt_embs = embeddings[neighbors]
        if args.metric == "gaussian":
            raw_dist = torch.cdist(embeddings[tid].unsqueeze(0), tgt_embs, p=2).pow(2).squeeze(0).cpu()
        else:
            src_norm = embeddings[tid] / embeddings[tid].norm().clamp(min=1e-10)
            tgt_norm = tgt_embs / tgt_embs.norm(dim=1, keepdim=True).clamp(min=1e-10)
            raw_dist = (1.0 - torch.mm(src_norm.unsqueeze(0), tgt_norm.t())).squeeze(0).cpu()

        print(f"\nTarget Token: '{actual_token}' (ID: {tid}, local_sigma: {kernel._sigma[tid].item():.4f} if VB)")
        print(f"{'Neighbor':<20} | {'Raw Dist':<10} | {'logR':<10} | {'Prob t=0.0':<12} | {'Prob t=0.5':<12}")
        print("-" * 110)

        for i in range(min(args.top_k, len(neighbors))):
            nb_id = neighbors[i].item()
            nb_str = tokenizer.decode([nb_id])
            lR = logR[i].item()
            rd = raw_dist[i].item()
            
            probs = []
            for t_val in [0.0, 0.5]:
                alpha = schedule.alpha_t(torch.tensor([t_val])).item()
                exponent = max(1e-10, alpha) ** args.beta
                logits = exponent * logR
                p = torch.softmax(logits, dim=0)[i].item()
                probs.append(p)
            
            print(f"'{nb_str}':{nb_id:<12} | {rd:8.4f}   | {lR:8.4f}   | {probs[0]:10.4f}   | {probs[1]:10.4f}")

    print("\n" + "="*110)
    print("ANALYSIS TIPS:")
    print("1. If 'Raw Dist' is small but 'Prob' is flat, VB (variable bandwidth) is too aggressive.")
    print("2. Try increasing --k_neighbors (e.g., 20) to make the local scale larger/stabler.")
    print("3. Try --variable_bandwidth False to see the 'Global' behavior.")
    print("="*110)

if __name__ == "__main__":
    main()
