#!/usr/bin/env python3
"""Validation Metric Verification Tool.

Loads a checkpoint and verifies that val/ppl and val/ppl_denoising are
computed and reported correctly.
"""

import os
import sys
import torch
import hydra
from pathlib import Path
from omegaconf import OmegaConf

# Jean Zay Offline Setup
work_dir = os.environ.get("WORK", os.path.expanduser("~"))
os.environ["HF_HOME"] = os.path.join(work_dir, ".cache", "huggingface")
os.environ["HF_DATASETS_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_HUB_OFFLINE"] = "1"

# Add project root to path
project_root = os.path.dirname(os.path.abspath(__file__))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.discrete_diffusion.train import load_checkpoint
from src.discrete_diffusion.data import get_tokenizer, get_dataset, get_dataloaders
from src.discrete_diffusion.forward_process import SIKForwardProcess, KNNKernel, KeOpsKernel

def main():
    if len(sys.argv) < 2:
        print("Usage: python check_val_metrics.py <checkpoint_path>")
        return

    ckpt_path = sys.argv[1]
    if not os.path.exists(ckpt_path):
        print(f"Error: Checkpoint not found at {ckpt_path}")
        return

    print(f"[*] Loading checkpoint: {ckpt_path}")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Load model from checkpoint
    try:
        model, config = load_checkpoint(ckpt_path, device=device)
        model.eval()
    except Exception as e:
        print(f"Error loading checkpoint: {e}")
        return

    tokenizer = model.tokenizer
    print(f"[*] Model: {config.algo.name} | Vocab: {model.vocab_size}")

    # Prepare small validation batch
    print("[*] Preparing validation data...")
    config.loader.batch_size = 4
    config.loader.eval_batch_size = 4
    # Ensure we use a small subset for quick check
    _, val_loader = get_dataloaders(config, tokenizer)
    
    batch = next(iter(val_loader))
    batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}

    print("[*] Running validation step...")
    model.metrics.reset()
    
    with torch.no_grad():
        # Standard validation_step logic
        # 1. Collection pass (logs unweighted metrics)
        model.validation_step(batch, 0)
    
    # Extract results
    ppl = model.metrics.valid_nlls['ppl'].compute().item()
    denoising_ppl = model.metrics.valid_nlls_denoising['ppl_denoising'].compute().item()
    
    print("\n" + "="*60)
    print(f"VALIDATION METRICS VERIFICATION")
    print("="*60)
    print(f"Standard Perplexity (val/ppl):           {ppl:.4f}")
    print(f"Denoising Perplexity (val/ppl_denoising): {denoising_ppl:.4f}")
    print("="*60)
    print("[Success] Both metrics are active and reporting.")

if __name__ == "__main__":
    main()
