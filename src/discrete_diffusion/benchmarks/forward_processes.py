#!/usr/bin/env python3
"""Benchmark script comparing SIK vs Absorbing forward processes."""

import argparse
import sys
import os
import time
import statistics
import json
from pathlib import Path

# Jean Zay Offline Setup
work_dir = os.environ.get("WORK", os.path.expanduser("~"))
hf_home = os.path.join(work_dir, ".cache", "huggingface")
os.environ["HF_HOME"] = hf_home
os.environ["HF_DATASETS_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_HUB_OFFLINE"] = "1"

import torch
import numpy as np
from transformers import GPT2Model

# Add project root to path
repo_root = Path(__file__).resolve().parents[3]
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from src.discrete_diffusion.forward_process import (
    AbsorbingForwardProcess,
    UniformForwardProcess,
    SIKForwardProcess,
)
from src.discrete_diffusion.forward_process.kernels import KNNKernel, KeOpsKernel
from src.discrete_diffusion.noise_schedules import LogLinear, sample_t
from src.discrete_diffusion.data import get_tokenizer
from omegaconf import OmegaConf


def benchmark_forward_process(forward_process, input_ids, t_mode="fixed", t_val=0.5, num_warmup=3, num_runs=10):
    """Benchmark a forward process on a batch."""
    device = input_ids.device
    batch_size = input_ids.shape[0]
    config = OmegaConf.create({"algo": {"t_eps": 1e-3}, "training": {"low_discrepancy_sampling": False}})

    def get_t():
        if t_mode == "fixed":
            return torch.full((batch_size,), t_val, device=device, dtype=torch.float32)
        else:
            return sample_t(config, batch_size, device=device)

    for _ in range(num_warmup):
        t = get_t()
        _ = forward_process(input_ids, t)
    
    if device.type == "cuda":
        torch.cuda.synchronize()
    
    times = []
    for _ in range(num_runs):
        t = get_t()
        if device.type == "cuda":
            torch.cuda.synchronize()
        start = time.perf_counter()
        _ = forward_process(input_ids, t)
        if device.type == "cuda":
            torch.cuda.synchronize()
        end = time.perf_counter()
        times.append(end - start)
    return times


def summarize_seed_means(seed_means_ms):
    """Summarize per-seed mean latencies in milliseconds."""
    mean_ms = statistics.mean(seed_means_ms)
    std_ms = statistics.stdev(seed_means_ms) if len(seed_means_ms) > 1 else 0.0
    median_ms = statistics.median(seed_means_ms)
    min_ms = min(seed_means_ms)
    max_ms = max(seed_means_ms)
    return {
        "mean": mean_ms,
        "std": std_ms,
        "median": median_ms,
        "min": min_ms,
        "max": max_ms,
        "num_seeds": len(seed_means_ms),
    }


def benchmark_across_seeds(
    forward_process,
    *,
    vocab_size,
    batch_size,
    seq_len,
    device,
    num_seeds,
    seed_base,
    t_mode="fixed",
    t_val=0.5,
    num_warmup=3,
    num_runs=10,
):
    """Benchmark a forward process across multiple random seeds."""
    seed_means_ms = []
    for seed_idx in range(num_seeds):
        seed = seed_base + seed_idx
        torch.manual_seed(seed)
        np.random.seed(seed % (2**32 - 1))
        if device.type == "cuda":
            torch.cuda.manual_seed_all(seed)
        input_ids = torch.randint(0, vocab_size, (batch_size, seq_len), device=device)
        times = benchmark_forward_process(
            forward_process,
            input_ids,
            t_mode=t_mode,
            t_val=t_val,
            num_warmup=num_warmup,
            num_runs=num_runs,
        )
        seed_means_ms.append(statistics.mean(times) * 1000)
    return summarize_seed_means(seed_means_ms)


def autotune_keops_config(
    tokenizer,
    schedule,
    embeddings,
    batch_input_ids,
    metric,
    args,
):
    """Pick a fast KeOps block configuration for the current hardware/problem size.

    This is intentionally a narrow sweep around the known-good exact dense settings.
    For one benchmark invocation, autotuning adds overhead. It pays off when you run
    the benchmark repeatedly or reuse the chosen configuration for real jobs.
    """
    cache_dir = Path(os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache")) / "gdds"
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path = cache_dir / "keops_autotune.json"
    autotune_version = "v3"
    cache_key = "|".join(
        [
            autotune_version,
            str(torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"),
            metric,
            str(args.batch_size),
            str(args.seq_len),
            str(len(tokenizer)),
            str(bool(args.keops_use_bf16)),
            str(bool(args.keops_use_cuda_sampler)),
        ]
    )

    if cache_path.exists():
        try:
            cache = json.loads(cache_path.read_text())
            if cache_key in cache:
                cfg = cache[cache_key]
                print(
                    "    [autotune] using cached config "
                    f"vocab={cfg['keops_vocab_block_size']}, "
                    f"unique={cfg['keops_unique_token_chunk_size']}, "
                    f"pos={cfg['keops_pos_chunk_size']}"
                )
                return cfg
        except Exception:
            pass

    if metric == "gaussian":
        # Gauss keeps improving with larger unique-token chunks. Also test
        # larger vocab blocks to reduce the number of block passes.
        vocab_candidates = [8192, 12288]
        unique_candidates = [16384, 24576, 32768]
        pos_candidates = [8192]
    else:
        # Cosine does not benefit from tiny vocab blocks; focus search on
        # larger blocks and larger unique-token chunks.
        vocab_candidates = [8192, 12288]
        unique_candidates = [8192, 16384, 24576]
        pos_candidates = [4096, 8192]
    candidates = [
        {
            "keops_pos_chunk_size": pos_chunk,
            "keops_vocab_block_size": vocab_block,
            "keops_unique_token_chunk_size": unique_chunk,
        }
        for vocab_block in vocab_candidates
        for unique_chunk in unique_candidates
        for pos_chunk in pos_candidates
    ]

    best_cfg = candidates[0]
    best_time = float("inf")
    print(f"    [autotune] searching {len(candidates)} KeOps configs for {metric}...")
    for cfg in candidates:
        k_params = {
            "embeddings": embeddings,
            "epsilon": 0.01,
            "gamma": 0.0,
            "metric": metric,
            "variable_bandwidth": True,
            "k_neighbors": 7,
        }
        kernel = KeOpsKernel(
            **k_params,
            pos_chunk_size=cfg["keops_pos_chunk_size"],
            vocab_block_size=cfg["keops_vocab_block_size"],
            unique_token_chunk_size=cfg["keops_unique_token_chunk_size"],
            use_bf16=args.keops_use_bf16,
            verbose=False,
            use_compiled_sampler=args.keops_use_compiled_sampler,
            use_triton_sampler=args.keops_use_triton_sampler,
            use_cuda_sampler=args.keops_use_cuda_sampler,
        )
        fp = SIKForwardProcess(tokenizer, schedule, kernel).to(batch_input_ids.device)
        trial_means = []
        for _ in range(3):
            times = benchmark_forward_process(
                fp,
                batch_input_ids,
                t_mode="training",
                num_warmup=1,
                num_runs=3,
            )
            trial_means.append(statistics.mean(times))
        mean = statistics.median(trial_means)
        print(
            "    [autotune] "
            f"vocab={cfg['keops_vocab_block_size']}, "
            f"unique={cfg['keops_unique_token_chunk_size']}, "
            f"pos={cfg['keops_pos_chunk_size']} -> {mean * 1000:.2f} ms "
            f"(trials: {', '.join(f'{t * 1000:.2f}' for t in trial_means)})"
        )
        if mean < best_time:
            best_time = mean
            best_cfg = cfg

    try:
        cache = {}
        if cache_path.exists():
            cache = json.loads(cache_path.read_text())
        cache[cache_key] = best_cfg
        cache_path.write_text(json.dumps(cache, indent=2, sort_keys=True))
    except Exception:
        pass

    print(
        "    [autotune] selected "
        f"vocab={best_cfg['keops_vocab_block_size']}, "
        f"unique={best_cfg['keops_unique_token_chunk_size']}, "
        f"pos={best_cfg['keops_pos_chunk_size']}"
    )
    return best_cfg


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--seq-len", type=int, default=1024)
    parser.add_argument("--num-warmup", type=int, default=3)
    parser.add_argument("--num-runs", type=int, default=10)
    parser.add_argument("--num-seeds", type=int, default=3)
    parser.add_argument("--seed-base", type=int, default=1234)
    # H100-tuned defaults for the exact dense CUDA sampler path. These are a
    # much better starting point than the older conservative chunk sizes.
    parser.add_argument("--keops-pos-chunk-size", type=int, default=4096)
    parser.add_argument("--keops-vocab-block-size", type=int, default=8192)
    parser.add_argument("--keops-unique-token-chunk-size", type=int, default=24576)
    parser.add_argument("--keops-use-bf16", dest="keops_use_bf16", action="store_true")
    parser.add_argument("--no-keops-use-bf16", dest="keops_use_bf16", action="store_false")
    parser.add_argument("--keops-verbose", action="store_true")
    parser.add_argument("--keops-use-compiled-sampler", action="store_true")
    parser.add_argument("--keops-use-triton-sampler", action="store_true")
    parser.add_argument("--keops-use-cuda-sampler", dest="keops_use_cuda_sampler", action="store_true")
    parser.add_argument("--no-keops-use-cuda-sampler", dest="keops_use_cuda_sampler", action="store_false")
    parser.add_argument("--keops-autotune", dest="keops_autotune", action="store_true")
    parser.add_argument("--no-keops-autotune", dest="keops_autotune", action="store_false")
    parser.set_defaults(
        keops_use_bf16=True,
        keops_use_cuda_sampler=True,
        keops_autotune=True,
    )
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[*] Device: {device}")

    # Setup
    config = OmegaConf.create({"data": {"tokenizer_name_or_path": "gpt2"}})
    tokenizer = get_tokenizer(config)
    model = GPT2Model.from_pretrained("gpt2", local_files_only=True)
    model.resize_token_embeddings(len(tokenizer))
    embeddings = model.wte.weight.detach().to(device)

    schedule = LogLinear(eps=1e-3)

    # Kernels to benchmark
    kernels = [
        {"metric": "gaussian", "impl": "knn", "name": "SIK Gauss (KNN)"},
        {"metric": "gaussian", "impl": "keops", "name": "SIK Gauss (KeOps)"},
        {"metric": "cosine", "impl": "knn", "name": "SIK Cosine (KNN)"},
        {"metric": "cosine", "impl": "keops", "name": "SIK Cosine (KeOps)"},
    ]

    # Batch used for autotuning only. Main benchmarking below runs across seeds.
    torch.manual_seed(args.seed_base)
    np.random.seed(args.seed_base % (2**32 - 1))
    if device == "cuda":
        torch.cuda.manual_seed_all(args.seed_base)
    batch_input_ids = torch.randint(0, len(tokenizer), (args.batch_size, args.seq_len), device=device)

    results = []
    
    # Standard Baselines
    print("\n[*] Benchmarking Baselines...")
    for fp_class, name in [(AbsorbingForwardProcess, "Absorbing"), (UniformForwardProcess, "Uniform")]:
        fp = fp_class(tokenizer, schedule).to(device)
        print(f"  {name}...", end=" ", flush=True)
        stats = benchmark_across_seeds(
            fp,
            vocab_size=len(tokenizer),
            batch_size=args.batch_size,
            seq_len=args.seq_len,
            device=batch_input_ids.device,
            num_seeds=args.num_seeds,
            seed_base=args.seed_base,
            t_mode="training",
            num_warmup=args.num_warmup,
            num_runs=args.num_runs,
        )
        print(f"{stats['mean']:.2f} +- {stats['std']:.2f} ms")
        results.append({"name": name, **stats})

    # SIK Variants
    print("\n[*] Benchmarking SIK Variants...")
    autotuned_cfgs = {}
    for k_cfg in kernels:
        print(f"  {k_cfg['name']}...", end=" ", flush=True)
        try:
            k_params = {
                "embeddings": embeddings,
                "epsilon": 0.01,
                "gamma": 0.0,
                "metric": k_cfg["metric"],
                "variable_bandwidth": True,
                "k_neighbors": 7,
            }
            if k_cfg["impl"] == "knn":
                kernel = KNNKernel(**k_params, top_k=64)
            else:
                tune_cfg = None
                if args.keops_autotune:
                    if k_cfg["metric"] not in autotuned_cfgs:
                        autotuned_cfgs[k_cfg["metric"]] = autotune_keops_config(
                            tokenizer,
                            schedule,
                            embeddings,
                            batch_input_ids,
                            k_cfg["metric"],
                            args,
                        )
                    tune_cfg = autotuned_cfgs[k_cfg["metric"]]
                kernel = KeOpsKernel(
                    **k_params,
                    pos_chunk_size=(
                        tune_cfg["keops_pos_chunk_size"] if tune_cfg is not None else args.keops_pos_chunk_size
                    ),
                    vocab_block_size=(
                        tune_cfg["keops_vocab_block_size"] if tune_cfg is not None else args.keops_vocab_block_size
                    ),
                    unique_token_chunk_size=(
                        tune_cfg["keops_unique_token_chunk_size"]
                        if tune_cfg is not None
                        else args.keops_unique_token_chunk_size
                    ),
                    use_bf16=args.keops_use_bf16,
                    verbose=args.keops_verbose,
                    use_compiled_sampler=args.keops_use_compiled_sampler,
                    use_triton_sampler=args.keops_use_triton_sampler,
                    use_cuda_sampler=args.keops_use_cuda_sampler,
                )
            
            fp = SIKForwardProcess(tokenizer, schedule, kernel).to(device)
            stats = benchmark_across_seeds(
                fp,
                vocab_size=len(tokenizer),
                batch_size=args.batch_size,
                seq_len=args.seq_len,
                device=batch_input_ids.device,
                num_seeds=args.num_seeds,
                seed_base=args.seed_base,
                t_mode="training",
                num_warmup=args.num_warmup,
                num_runs=args.num_runs,
            )
            print(f"{stats['mean']:.2f} +- {stats['std']:.2f} ms")
            results.append({"name": k_cfg["name"], **stats})
        except Exception as e:
            print(f"FAILED: {e}")

    # Summary
    print("\n" + "="*60)
    print(
        f"{'Method':<25} | {'Mean +- Std (ms)':<18} | "
        f"{'Median':>8} | {'Min':>8} | {'Max':>8} | {'Seeds':>5}"
    )
    print("-" * 60)
    for r in results:
        print(
            f"{r['name']:<25} | "
            f"{r['mean']:>7.2f} +- {r['std']:<7.2f} | "
            f"{r['median']:>8.2f} | "
            f"{r['min']:>8.2f} | "
            f"{r['max']:>8.2f} | "
            f"{r['num_seeds']:>5}"
        )
    print("=" * 60)

if __name__ == "__main__":
    main()
