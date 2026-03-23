#!/usr/bin/env python3
print(">>> Animation script starting (Booting Python interpreter)...")

import os
import sys
import time
from pathlib import Path

print(">>> Importing core libraries (torch, numpy)...")
import argparse
import torch
import numpy as np
from omegaconf import OmegaConf

# ANSI Colors
RESET = "\033[0m"
BOLD = "\033[1m"
GREEN = "\033[32m"
CYAN = "\033[36m"
YELLOW = "\033[33m"
GRAY = "\033[90m"
BLUE = "\033[34m"

# Add project root to path
repo_root = Path(__file__).resolve().parents[3]
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

print(">>> Loading GDDS components (SIK, Schedules, Tokenizers)...")
from src.discrete_diffusion.forward_process import SIKForwardProcess, KNNKernel
from src.discrete_diffusion.noise_schedules import LogLinear
from src.discrete_diffusion.data import get_tokenizer

def render_lambda_plot(sik, t_curr, width=60, height=8):
    """ASCII plot of λ(t) with a marker for current time."""
    t_grid = np.linspace(0.0, 1.0, width)
    t_tensor = torch.tensor(t_grid, dtype=torch.float32)
    with torch.no_grad():
        lam_grid = sik._lambda(t_tensor).numpy()
    
    canvas = [[' ' for _ in range(width)] for _ in range(height)]
    
    # Plot the curve
    for x_idx, l_val in enumerate(lam_grid):
        y_idx = int(l_val * (height - 1))
        canvas[height - 1 - y_idx][x_idx] = '·'
        
    # Add current time marker
    curr_x = int(t_curr * (width - 1))
    curr_y = int(lam_grid[curr_x] * (height - 1))
    canvas[height - 1 - curr_y][curr_x] = YELLOW + '█' + RESET
    
    lines = []
    lines.append(f" {GRAY}λ(t){RESET} |")
    for i, row in enumerate(canvas):
        prefix = " 1.0 |" if i == 0 else "     |"
        lines.append(prefix + "".join(row))
    lines.append(f" 0.0 |" + "_" * width)
    
    # Timeline
    timeline = [" "] * width
    timeline[0] = "0"
    timeline[width//2] = "0.5"
    timeline[width-1] = "1"
    lines.append("      " + "".join(timeline) + f" {GRAY}Time t{RESET}")
    
    return "\n".join(lines)

def main():
    parser = argparse.ArgumentParser(description="Animate SIK noising.")
    parser.add_argument("--text", type=str, default="The cat sat on the mat", help="Text to noise")
    parser.add_argument("--fps", type=int, default=10, help="Frames per second")
    parser.add_argument("--steps", type=int, default=150, help="Total animation steps")
    parser.add_argument("--t0", type=float, default=0.6, help="Time where teleportation hits 50%")
    parser.add_argument("--slope", type=float, default=10.0, help="Sigmoid slope for teleportation transition")
    args = parser.parse_args()

    cache_path = repo_root / "gpt2_sik_cache.pt"
    if not cache_path.exists():
        print(f"\nError: Cache '{cache_path}' not found.")
        return

    cache = torch.load(cache_path, map_location="cpu")
    tokenizer = get_tokenizer(OmegaConf.create({"data": {"tokenizer_name_or_path": "gpt2"}}))
    schedule = LogLinear(eps=1e-3)
    
    class CachedKNNKernel(KNNKernel):
        def __init__(self, cache_data):
            torch.nn.Module.__init__(self)
            self._knn_indices = cache_data["knn_indices"]
            self._logR_vocab = cache_data["logR_vocab"]
            self._sigma = cache_data["sigma"]
            self.m = cache_data["vocab_size"]
            self.device = self._knn_indices.device
            self.epsilon = 0.01 
            self.variable_bandwidth = True

    mock_kernel = CachedKNNKernel(cache)
    sik = SIKForwardProcess(
        tokenizer=tokenizer, schedule=schedule, kernel=mock_kernel,
        temperature_beta=0.0, lambda_min=0.01,
        lambda_sigmoid_s=args.slope, lambda_t0=args.t0
    )

    input_ids = tokenizer.encode(args.text, add_special_tokens=False, return_tensors="pt")
    _, info = sik(input_ids, torch.tensor([1.0]), return_info=True, return_history=True)
    
    if isinstance(info['history'], list):
        history_arr = torch.stack(info['history'], dim=-1)[0].cpu().numpy()
    else:
        history_arr = info['history'][0].cpu().numpy()

    jump_times = info['jump_times'][0].cpu().numpy()
    seq_len = input_ids.shape[1]
    
    print(f"\n{BOLD}>>> ALL SYSTEMS GO! Starting animation...{RESET}")
    time.sleep(1)

    try:
        sys.stdout.write("\033[2J") # Initial Clear

        for step in range(args.steps + 1):
            t_curr = step / args.steps
            
            sys.stdout.write("\033[H") # Home
            sys.stdout.write(f"{BOLD}SIK FORWARD NOISING ANIMATION{RESET}\n")
            sys.stdout.write("=" * 115 + "\n")
            
            t_tensor = torch.tensor([t_curr], dtype=torch.float32)
            p_tele = sik._lambda(t_tensor).item()
            
            # Dashboard Header
            meter_len = 30
            p_filled = int(p_tele * meter_len)
            meter = YELLOW + "█" * p_filled + GRAY + "-" * (meter_len - p_filled) + RESET
            sys.stdout.write(f"{CYAN}Time t: {t_curr:.2f}{RESET} | Teleport Prob λ(t): [{meter}] {p_tele*100:3.0f}%\n")
            
            total_jumps = np.sum(jump_times <= t_curr)
            sys.stdout.write(f"Progress: {step}/{args.steps} steps | Total Jumps in Seq: {total_jumps}\n")
            sys.stdout.write("-" * 115 + "\n\n")

            sys.stdout.write(f"{BOLD}POSITION-WISE TRANSITION HISTORY ({BLUE}·{RESET} = leading space):{RESET}\n")
            
            current_tokens = []
            for i in range(seq_len):
                j_idx = np.sum(jump_times[i] <= t_curr)
                tokens_so_far = history_arr[i, 0 : j_idx + 1]
                times_so_far = jump_times[i, 0 : j_idx]
                current_tokens.append(tokens_so_far[-1])
                
                token_strs = []
                for idx, tid in enumerate(tokens_so_far):
                    t_raw = tokenizer.convert_ids_to_tokens(int(tid))
                    
                    if idx == 0:
                        display = t_raw.replace('Ġ', f"{BLUE}·{RESET}{BOLD}")
                        token_strs.append(f"{BOLD}{display}{RESET}")
                    else:
                        prev_tid = int(tokens_so_far[idx-1])
                        prev_raw = tokenizer.convert_ids_to_tokens(prev_tid)
                        if t_raw == prev_raw:
                            t_raw = f"{t_raw}[{tid}]"
                        
                        is_kernel = int(tid) in mock_kernel._knn_indices[prev_tid]
                        color = GREEN if is_kernel else YELLOW
                        display = t_raw.replace('Ġ', f"{BLUE}·{RESET}{color}")
                        j_t = times_so_far[idx-1]
                        token_strs.append(f"{GRAY}({j_t:.2f}){RESET} {color}{display}{RESET}")
                
                history_line = f" {CYAN}→{RESET} ".join(token_strs)
                sys.stdout.write(f"Pos {i:2}: {history_line}\033[K\n")
            
            sys.stdout.write("\n" + "-" * 115 + "\n")
            summary = tokenizer.decode(current_tokens).replace('\n', ' ')
            sys.stdout.write(f"{BOLD}CURRENT SEQ:{RESET}  {summary}\033[K\n")
            sys.stdout.write("-" * 115 + "\n")
            
            # Add Lambda Plot at the very bottom
            sys.stdout.write(f"{BOLD}Teleportation Schedule Profile:{RESET}\n")
            sys.stdout.write(render_lambda_plot(sik, t_curr) + "\n")
            
            sys.stdout.write(f"{GRAY}Legend: {GREEN}Kernel (Semantic) Jump{RESET} | {YELLOW}Random (Uniform) Jump{RESET} | {GRAY}Slope={args.slope}, t0={args.t0}{RESET}\n")
            sys.stdout.write("-" * 115 + "\n")
            sys.stdout.flush()
            
            time.sleep(1.0 / args.fps)
            
    except KeyboardInterrupt:
        print("\nAnimation stopped.")

if __name__ == "__main__":
    main()
