#!/usr/bin/env python3
"""Visualize SIK forward process jump-by-jump with the k-NN kernel at t=1.0."""

import argparse
import os
import sys
from pathlib import Path

# CRITICAL: Set environment variables BEFORE importing transformers
work_dir = os.environ.get("WORK", os.path.expanduser("~"))
hf_home = os.path.join(work_dir, ".cache", "huggingface")
os.environ["HF_HOME"] = hf_home
os.environ["HF_DATASETS_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_HUB_OFFLINE"] = "1"

import torch
from omegaconf import OmegaConf
from transformers import GPT2Model

# Add project root to path
repo_root = Path(__file__).resolve().parents[3]
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from src.discrete_diffusion.forward_process import SIKForwardProcess, KNNKernel
from src.discrete_diffusion.noise_schedules import LogLinear
from src.discrete_diffusion.data import get_tokenizer


def main():
    parser = argparse.ArgumentParser(description="Visualize SIK noising with the k-NN kernel.")
    parser.add_argument("--sentence", type=str, default="The cat sat on the mat")
    parser.add_argument("--metric", type=str, default="gaussian", choices=["gaussian", "cosine"])
    parser.add_argument("--epsilon", type=float, default=0.01)
    parser.add_argument("--beta", type=float, default=0.0)
    parser.add_argument("--gamma", type=float, default=0.0)
    parser.add_argument("--top_k", type=int, default=64)
    parser.add_argument("--k_neighbors", type=int, default=7)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[*] Device: {device}")

    tokenizer = get_tokenizer(OmegaConf.create({"data": {"tokenizer_name_or_path": "gpt2"}}))
    model = GPT2Model.from_pretrained("gpt2")
    model.resize_token_embeddings(len(tokenizer))
    embeddings = model.wte.weight.detach().to(device)

    print(f"[*] Building SIK k-NN kernel ({args.metric})...")
    kernel = KNNKernel(
        embeddings=embeddings,
        epsilon=args.epsilon,
        gamma=args.gamma,
        metric=args.metric,
        variable_bandwidth=True,
        k_neighbors=args.k_neighbors,
        top_k=args.top_k,
    )

    schedule = LogLinear(eps=1e-3)
    sik = SIKForwardProcess(
        tokenizer=tokenizer,
        schedule=schedule,
        kernel=kernel,
        temperature_beta=args.beta,
        lambda_min=0.01,
        lambda_sigmoid_s=5.0,
        lambda_t0=0.4,
    ).to(device)

    tokens = tokenizer.encode(args.sentence, add_special_tokens=False)
    input_ids = torch.tensor([tokens], device=device)
    batch_size, seq_len = input_ids.shape

    x_t, info = sik(input_ids, torch.ones(batch_size, device=device), return_info=True, return_history=True)

    history = info["history"]
    jump_times = info["jump_times"]
    num_jumps = info["num_jumps"][0].cpu().tolist()

    print("\n" + "=" * 72)
    print("Sequence Evolution (SIK + k-NN Kernel)")
    print("=" * 72)

    time_snapshots = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]

    print(f"\n{'Time':>6} | Sequence")
    print(f"{'-'*6}-|-{'-'*58}")

    for t_query in time_snapshots:
        snapshot_tokens = []
        for pos in range(seq_len):
            nj = num_jumps[pos]
            if nj == 0:
                snapshot_tokens.append(tokens[pos])
                continue

            pos_jump_times = jump_times[0, pos, :nj].cpu()
            jumps_before = (pos_jump_times < t_query).sum().item()
            token_id = history[jumps_before][0, pos].item()
            snapshot_tokens.append(token_id)

        seq = " ".join(
            tokenizer.decode([tid], clean_up_tokenization_spaces=False)
            for tid in snapshot_tokens
        )
        print(f"{t_query:6.1f} | {seq}")

    print(f"\n[*] Final sequence: {tokenizer.decode(x_t[0].tolist(), clean_up_tokenization_spaces=False)}")


if __name__ == "__main__":
    main()
