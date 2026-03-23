#!/usr/bin/env python3
print(">>> Cache builder starting (Booting Python interpreter)...")

import os
import sys
import time
from pathlib import Path

print(">>> Importing libraries (torch, transformers, omegaconf)...")
import torch
import numpy as np
from omegaconf import OmegaConf

# Jean Zay Offline Setup
work_dir = os.environ.get("WORK", os.path.expanduser("~"))
hf_home = os.path.join(work_dir, ".cache", "huggingface")
os.environ["HF_HOME"] = hf_home
os.environ["HF_DATASETS_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_HUB_OFFLINE"] = "1"

from transformers import GPT2Model

# Add project root to path
repo_root = Path(__file__).resolve().parents[3]
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from src.discrete_diffusion.data import get_tokenizer
from src.discrete_diffusion.forward_process import KNNKernel


def load_gpt2_embeddings(tokenizer):
    try:
        model = GPT2Model.from_pretrained("gpt2", local_files_only=True)
    except (OSError, AttributeError) as exc:
        raise RuntimeError(
            "GPT-2 model weights were not found in the local Hugging Face cache. "
            "This cache builder runs in offline mode, so cache GPT-2 first."
        ) from exc

    model.resize_token_embeddings(len(tokenizer))
    return model.wte.weight.detach()

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\n[*] BUILDING SIK CACHE ON DEVICE: {device}")
    print("=" * 60)

    print("[*] Step 1/4: Loading tokenizer...")
    tokenizer = get_tokenizer(OmegaConf.create({"data": {"tokenizer_name_or_path": "gpt2"}}))
    
    print("[*] Step 2/4: Loading GPT-2 model and resizing embeddings...")
    embeddings = load_gpt2_embeddings(tokenizer)

    print("[*] Step 3/4: Building k-NN graph (This is the slow part)...")
    print("    [Info] Using parameters: epsilon=0.01, gamma=0.0, k=64, top_k=64")
    
    start_time = time.perf_counter()
    kernel = KNNKernel(
        embeddings=embeddings.to(device), epsilon=0.01, gamma=0.0, metric="gaussian",
        variable_bandwidth=True, k_neighbors=7, top_k=64
    )
    duration = time.perf_counter() - start_time
    print(f"    [Success] Kernel construction took {duration:.2f} seconds.")

    print("[*] Step 4/4: Serializing data to 'gpt2_sik_cache.pt'...")
    data = {
        "embeddings": embeddings.cpu(),
        "knn_indices": kernel._knn_indices.cpu(),
        "logR_vocab": kernel._logR_vocab.cpu(),
        "sigma": kernel._sigma.cpu(),
        "vocab_size": len(tokenizer)
    }

    # Save to the project root
    save_path = repo_root / "gpt2_sik_cache.pt"
    torch.save(data, save_path)
    
    print("=" * 60)
    print(f"{BOLD}CACHE CONSTRUCTION COMPLETE!{RESET}")
    print(f"File size: {os.path.getsize(save_path) / 1024 / 1024:.2f} MB")
    print(f"Location: {save_path}")
    print("You can now run 'python src/discrete_diffusion/visualizations/animate_sik.py' instantly.")
    print("=" * 60)

if __name__ == "__main__":
    BOLD = "\033[1m"
    RESET = "\033[0m"
    main()
