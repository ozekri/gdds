#!/usr/bin/env python3
print(">>> KeOps animation script starting (Booting Python interpreter)...")

import os
import sys
import time
from pathlib import Path

# CRITICAL: Set environment variables BEFORE importing transformers
work_dir = os.environ.get("WORK", os.path.expanduser("~"))
hf_home = os.path.join(work_dir, ".cache", "huggingface")
os.environ["HF_HOME"] = hf_home
os.environ["HF_DATASETS_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_HUB_OFFLINE"] = "1"

print(">>> Importing core libraries (torch, numpy)...")
import argparse
import numpy as np
import torch
from omegaconf import OmegaConf
from transformers import GPT2Model

# ANSI Colors
RESET = "\033[0m"
BOLD = "\033[1m"
CYAN = "\033[36m"
GRAY = "\033[90m"
BLUE = "\033[34m"


def gradient_color(t_value: float) -> str:
    """Green -> yellow -> red ANSI 256-color gradient as noise time increases."""
    t_clamped = max(0.0, min(1.0, float(t_value)))
    if t_clamped <= 0.5:
        frac = t_clamped / 0.5
        r, g, b = int(255 * frac), 255, 0
    else:
        frac = (t_clamped - 0.5) / 0.5
        r, g, b = 255, int(255 * (1.0 - frac)), 0
    return f"\033[38;2;{r};{g};{b}m"


def token_cell(tokenizer, token_id: int, width: int = 10) -> str:
    raw = tokenizer.convert_ids_to_tokens(int(token_id))
    text = raw.replace("Ġ", "·")
    if len(text) > width:
        text = text[: width - 1] + "…"
    return text.ljust(width)


def colored_cell(text: str, color: str | None) -> str:
    if not color:
        return text
    return f"{color}{text}{RESET}"


def write_line(text: str = "") -> None:
    sys.stdout.write(text + "\033[K\n")


def write_block(text: str) -> None:
    for line in text.splitlines():
        write_line(line)


def time_cell(value: float | None, width: int = 9) -> str:
    if value is None:
        return "-".center(width)
    return f"{value:0.2f}".center(width)


# Add project root to path
repo_root = Path(__file__).resolve().parents[3]
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

print(">>> Loading GDDS components (KeOps, CTMC, Schedules, Tokenizers)...")
from src.discrete_diffusion.forward_process import BaseCTMCForwardProcess, KeOpsKernel
from src.discrete_diffusion.noise_schedules import LogLinear
from src.discrete_diffusion.data import get_tokenizer


class SemanticKeOpsForwardProcess(BaseCTMCForwardProcess):
    """Pure semantic CTMC driven only by the KeOps kernel."""

    def __init__(self, tokenizer, schedule, kernel, *, temperature_beta: float = 0.0):
        super().__init__(tokenizer=tokenizer, schedule=schedule, name="semantic_keops")
        self.kernel = kernel
        self.temperature_beta = temperature_beta

    def _exponent(self, alpha_t: torch.Tensor) -> torch.Tensor:
        return alpha_t.pow(self.temperature_beta)

    def transition_kernel(self, curr_tokens: torch.Tensor, tk: torch.Tensor) -> torch.Tensor:
        alpha_tk = self.schedule.alpha_t(tk)
        exponent = self._exponent(alpha_tk)
        return self.kernel.sample_neighbors(curr_tokens, exponent)


def load_gpt2_embeddings(tokenizer, device: str) -> torch.Tensor:
    model = GPT2Model.from_pretrained("gpt2")
    model.resize_token_embeddings(len(tokenizer))
    return model.wte.weight.detach().to(device)


def render_lambda_plot(schedule, t_curr, snapshot_t, width=60, height=8):
    """ASCII plot of Λ(t) = -log(alpha(t)) with markers for current and snapshot times."""
    t_grid = np.linspace(0.0, 1.0, width)
    t_tensor = torch.tensor(t_grid, dtype=torch.float32)
    with torch.no_grad():
        alpha_grid = schedule.alpha_t(t_tensor).clamp(min=1e-10)
        lambda_grid = (-torch.log(alpha_grid)).cpu().numpy()
    lambda_max = max(float(lambda_grid.max()), 1e-8)

    canvas = [[" " for _ in range(width)] for _ in range(height)]
    for x_idx, lam_val in enumerate(lambda_grid):
        y_idx = int((lam_val / lambda_max) * (height - 1))
        canvas[height - 1 - y_idx][x_idx] = "·"

    curr_x = int(t_curr * (width - 1))
    curr_y = int((lambda_grid[curr_x] / lambda_max) * (height - 1))
    snap_x = int(snapshot_t * (width - 1))
    snap_y = int((lambda_grid[snap_x] / lambda_max) * (height - 1))

    canvas[height - 1 - snap_y][snap_x] = CYAN + "◆" + RESET
    canvas[height - 1 - curr_y][curr_x] = gradient_color(t_curr) + "█" + RESET

    lines = []
    lines.append(f" {GRAY}Λ(t) = -log(α_t){RESET}")
    lines.append("       +" + "-" * width + "+")
    for i, row in enumerate(canvas):
        if i == 0:
            prefix = f"{lambda_max:5.2f} |"
        elif i == height - 1:
            prefix = f"{0.00:5.2f} |"
        else:
            prefix = "      |"
        lines.append(prefix + "".join(row) + "|")
    lines.append("       +" + "-" * width + "+")

    timeline = [" "] * width
    timeline[0] = "0"
    timeline[width // 2] = "0.5"
    timeline[width - 1] = "1"
    lines.append("        " + "".join(timeline) + f" {GRAY}time t{RESET}")
    return "\n".join(lines)


def state_at_time(history_arr, jump_times, pos: int, t_query: float) -> tuple[int, float | None]:
    """Return token id and the jump time that produced it, if any."""
    j_idx = int(np.sum(jump_times[pos] <= t_query))
    token_id = int(history_arr[pos, j_idx])
    change_time = None if j_idx == 0 else float(jump_times[pos, j_idx - 1])
    return token_id, change_time


def first_last_change_times(jump_times, pos: int, t_query: float) -> tuple[float | None, float | None, int]:
    valid_times = jump_times[pos][jump_times[pos] <= t_query]
    if valid_times.size == 0:
        return None, None, 0
    return float(valid_times[0]), float(valid_times[-1]), int(valid_times.size)


def format_snapshot_sequence(tokenizer, snapshot_tokens, initial_tokens, snapshot_change_times):
    parts = []
    for tid, initial_tid, change_time in zip(snapshot_tokens, initial_tokens, snapshot_change_times):
        piece = tokenizer.decode([int(tid)], clean_up_tokenization_spaces=False)
        if change_time is None:
            parts.append(piece)
        else:
            parts.append(f"{gradient_color(change_time)}{piece}{RESET}")
    return "".join(parts)


def format_colored_sequence(tokenizer, token_ids, change_times):
    parts = []
    for tid, change_time in zip(token_ids, change_times):
        piece = tokenizer.decode([int(tid)], clean_up_tokenization_spaces=False)
        if change_time is None:
            parts.append(piece)
        else:
            parts.append(f"{gradient_color(change_time)}{piece}{RESET}")
    return "".join(parts)


def render_sequence_cells(tokenizer, token_ids, change_times, label: str, max_positions: int = 12, cell_width: int = 9) -> str:
    shown = min(len(token_ids), max_positions)
    cells = []
    for i in range(shown):
        text = token_cell(tokenizer, int(token_ids[i]), cell_width)
        color = None if change_times[i] is None else gradient_color(change_times[i] or 0.0)
        cells.append(colored_cell(text, color))
    line = f"{label:<10} | " + " ".join(cells)
    if len(token_ids) > shown:
        line += f" {GRAY}... ({shown}/{len(token_ids)} shown){RESET}"
    return line


def render_token_comparison(
    tokenizer,
    initial_tokens,
    current_tokens,
    current_change_times,
    current_first_change_times,
    current_jump_counts,
    current_status_cells,
    snapshot_tokens,
    snapshot_change_times,
    snapshot_status_cells,
    max_positions: int = 12,
    cell_width: int = 9,
) -> str:
    shown = min(len(initial_tokens), max_positions)
    label_width = 10

    def row(label: str, cells: list[str]) -> str:
        return f"{label:<{label_width}} | " + " ".join(cells)

    pos_cells = [str(i).center(cell_width) for i in range(shown)]
    init_cells = [token_cell(tokenizer, initial_tokens[i], cell_width) for i in range(shown)]
    curr_cells = []
    snap_cells = []
    jump_cells = []
    first_cells = []
    last_cells = []

    for i in range(shown):
        curr_text = token_cell(tokenizer, current_tokens[i], cell_width)
        curr_color = None if current_change_times[i] is None else gradient_color(current_change_times[i] or 0.0)
        curr_cells.append(colored_cell(curr_text, curr_color))
        jump_cells.append(str(current_jump_counts[i]).center(cell_width))
        first_cells.append(time_cell(current_first_change_times[i], cell_width))
        last_cells.append(time_cell(current_change_times[i], cell_width))

        snap_text = token_cell(tokenizer, snapshot_tokens[i], cell_width)
        snap_color = None if snapshot_change_times[i] is None else gradient_color(snapshot_change_times[i] or 0.0)
        snap_cells.append(colored_cell(snap_text, snap_color))

    lines = [
        f"{BOLD}STATE SUMMARY{RESET}",
        row("pos", pos_cells),
        row("now state", current_status_cells[:shown]),
        row("snap state", snapshot_status_cells[:shown]),
        row("jumps", jump_cells),
        row("first", first_cells),
        row("last", last_cells),
        "",
        f"{BOLD}TOKEN VIEW{RESET}",
        row("pos", pos_cells),
        row("x0", init_cells),
        row("now", curr_cells),
        row("snap", snap_cells),
    ]
    if len(initial_tokens) > shown:
        lines.append(f"{GRAY}Showing first {shown}/{len(initial_tokens)} positions{RESET}")
    lines.append(
        f"{GRAY}State legend: same = unchanged, back = jumped and returned to x0, diff = different from x0{RESET}"
    )
    return "\n".join(lines)


def render_position_history_line(tokenizer, tokens_so_far, times_so_far, max_hops: int) -> str:
    token_strs = []
    total_hops = len(tokens_so_far) - 1
    if total_hops > max_hops:
        start_idx = total_hops - max_hops + 1
    else:
        start_idx = 0

    for idx in range(len(tokens_so_far)):
        tid = int(tokens_so_far[idx])
        t_raw = tokenizer.convert_ids_to_tokens(tid)

        if idx == 0:
            display = t_raw.replace("Ġ", f"{BLUE}·{RESET}{BOLD}")
            token_strs.append(f"{BOLD}{display}{RESET}")
            if total_hops > max_hops:
                token_strs.append(f"{GRAY}...{RESET}")
        else:
            if idx < start_idx:
                continue
            prev_tid = int(tokens_so_far[idx - 1])
            prev_raw = tokenizer.convert_ids_to_tokens(prev_tid)
            if t_raw == prev_raw:
                t_raw = f"{t_raw}[{tid}]"
            j_t = float(times_so_far[idx - 1])
            color = gradient_color(j_t)
            display = t_raw.replace("Ġ", f"{BLUE}·{RESET}{color}")
            token_strs.append(f"{GRAY}({j_t:.2f}){RESET} {color}{display}{RESET}")
    return f" {CYAN}→{RESET} ".join(token_strs)


def render_jump_histogram(jump_times, bins: int = 24, width: int = 48) -> str:
    valid = jump_times[np.isfinite(jump_times)]
    if valid.size == 0:
        return f"{BOLD}JUMP-TIME HISTOGRAM{RESET}\n(no jumps)"

    counts, edges = np.histogram(valid, bins=bins, range=(0.0, 1.0))
    max_count = max(int(counts.max()), 1)
    bar_line = []
    for idx, count in enumerate(counts):
        if count == 0:
            bar_line.append(" ")
        else:
            t_mid = 0.5 * (edges[idx] + edges[idx + 1])
            bar_line.append(f"{gradient_color(t_mid)}█{RESET}")
    bar = "".join(bar_line)
    return "\n".join([
        f"{BOLD}JUMP-TIME HISTOGRAM{RESET}",
        f"{max_count:>4} | {bar}",
        f"{0:>4} | " + "-" * bins,
        "      0" + " " * max(1, bins // 2 - 1) + "0.5" + " " * max(1, bins // 2 - 2) + "1",
    ])


def status_cell(curr_token: int, initial_token: int, jump_count: int, width: int = 9) -> str:
    if jump_count == 0:
        label = "same"
    elif int(curr_token) == int(initial_token):
        label = "back"
    else:
        label = "diff"
    return label.center(width)


def main():
    parser = argparse.ArgumentParser(description="Animate semantic KeOps noising.")
    parser.add_argument("--text", type=str, default="The cat sat on the mat", help="Text to noise")
    parser.add_argument("--fps", type=int, default=10, help="Frames per second")
    parser.add_argument("--steps", type=int, default=150, help="Total animation steps")
    parser.add_argument("--metric", type=str, default="gaussian", choices=["gaussian", "cosine"])
    parser.add_argument("--epsilon", type=float, default=0.01)
    parser.add_argument("--beta", type=float, default=1.0, help="Temperature beta")
    parser.add_argument("--gamma", type=float, default=0.0)
    parser.add_argument("--k_neighbors", type=int, default=7)
    parser.add_argument("--keops-pos-chunk-size", type=int, default=2048)
    parser.add_argument("--keops-vocab-block-size", type=int, default=2048)
    parser.add_argument("--keops-unique-token-chunk-size", type=int, default=4096)
    parser.add_argument("--keops-use-bf16", action="store_true")
    parser.add_argument("--keops-verbose", action="store_true")
    parser.add_argument("--keops-use-compiled-sampler", action="store_true")
    parser.add_argument("--keops-use-triton-sampler", action="store_true")
    parser.add_argument("--keops-use-cuda-sampler", action="store_true")
    parser.add_argument("--max-positions-shown", type=int, default=12, help="Maximum positions shown in summary tables.")
    parser.add_argument("--max-hops-shown", type=int, default=6, help="Maximum recent hops shown per position in the history panel.")
    parser.add_argument("--focus-pos", type=int, default=None, help="If set, only expand one position's detailed history.")
    parser.add_argument("--compact", action="store_true", help="Show fewer detailed position histories each frame.")
    parser.add_argument(
        "--snapshot-time",
        type=float,
        default=None,
        help="Fixed snapshot time in [0, 1]. If omitted, a random snapshot time is sampled once per run.",
    )
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    tokenizer = get_tokenizer(OmegaConf.create({"data": {"tokenizer_name_or_path": "gpt2"}}))
    embeddings = load_gpt2_embeddings(tokenizer, device)
    schedule = LogLinear(eps=1e-3)

    kernel = KeOpsKernel(
        embeddings=embeddings,
        epsilon=args.epsilon,
        gamma=args.gamma,
        metric=args.metric,
        variable_bandwidth=True,
        k_neighbors=args.k_neighbors,
        pos_chunk_size=args.keops_pos_chunk_size,
        vocab_block_size=args.keops_vocab_block_size,
        unique_token_chunk_size=args.keops_unique_token_chunk_size,
        use_bf16=args.keops_use_bf16,
        verbose=args.keops_verbose,
        use_compiled_sampler=args.keops_use_compiled_sampler,
        use_triton_sampler=args.keops_use_triton_sampler,
        use_cuda_sampler=args.keops_use_cuda_sampler,
    )
    semantic_fp = SemanticKeOpsForwardProcess(
        tokenizer=tokenizer,
        schedule=schedule,
        kernel=kernel,
        temperature_beta=args.beta,
    )

    input_ids = tokenizer.encode(args.text, add_special_tokens=False, return_tensors="pt").to(device)
    _, info = semantic_fp(input_ids, torch.tensor([1.0], device=device), return_info=True, return_history=True)

    if isinstance(info["history"], list):
        history_arr = torch.stack(info["history"], dim=-1)[0].cpu().numpy()
    else:
        history_arr = info["history"][0].cpu().numpy()

    jump_times = info["jump_times"][0].cpu().numpy()
    seq_len = input_ids.shape[1]
    initial_tokens = input_ids[0].cpu().numpy()
    if args.snapshot_time is not None:
        if not 0.0 <= args.snapshot_time <= 1.0:
            raise ValueError("--snapshot-time must be in [0, 1].")
        snapshot_t = float(args.snapshot_time)
    else:
        snapshot_t = float(np.random.uniform(0.0, 1.0))
    snapshot_tokens = []
    snapshot_change_times = []
    for pos in range(seq_len):
        token_id, change_time = state_at_time(history_arr, jump_times, pos, snapshot_t)
        snapshot_tokens.append(token_id)
        snapshot_change_times.append(change_time)
    snapshot_sequence = format_snapshot_sequence(
        tokenizer,
        snapshot_tokens,
        initial_tokens,
        snapshot_change_times,
    )
    snapshot_first_change_times = []
    snapshot_jump_counts = []
    snapshot_status_cells = []
    for pos in range(seq_len):
        first_change, _, jump_count = first_last_change_times(jump_times, pos, snapshot_t)
        snapshot_first_change_times.append(first_change)
        snapshot_jump_counts.append(jump_count)
        snapshot_status_cells.append(status_cell(snapshot_tokens[pos], initial_tokens[pos], jump_count))

    if args.focus_pos is not None and not 0 <= args.focus_pos < seq_len:
        raise ValueError(f"--focus-pos must be between 0 and {seq_len - 1}.")

    print(f"\n{BOLD}>>> ALL SYSTEMS GO! Starting KeOps animation...{RESET}")
    time.sleep(1)

    try:
        sys.stdout.write("\033[2J")

        for step in range(args.steps + 1):
            t_curr = step / args.steps

            sys.stdout.write("\033[H")
            write_line(f"{BOLD}SEMANTIC KEOPS FORWARD NOISING ANIMATION{RESET}")
            write_line("=" * 115)

            t_tensor = torch.tensor([t_curr], dtype=torch.float32, device=device)
            alpha_t = schedule.alpha_t(t_tensor).item()
            lambda_t = float(-torch.log(schedule.alpha_t(t_tensor).clamp(min=1e-10)).item())

            meter_len = 30
            noise_level = 1.0 - alpha_t
            n_filled = int(noise_level * meter_len)
            meter_color = gradient_color(t_curr)
            meter = meter_color + "█" * n_filled + GRAY + "-" * (meter_len - n_filled) + RESET
            write_line(
                f"{CYAN}Time t: {t_curr:.2f}{RESET} | Λ(t)=-log(α): {lambda_t:5.2f} | Noise 1-α(t): [{meter}] {noise_level*100:3.0f}%"
            )

            total_jumps = np.sum(jump_times <= t_curr)
            write_line(f"Progress: {step}/{args.steps} steps | Total Jumps in Seq: {total_jumps}")
            write_line("-" * 115)
            write_line()

            write_line(
                f"{BOLD}POSITION-WISE TRANSITION HISTORY ({BLUE}·{RESET} = leading space):{RESET}"
            )

            current_tokens = []
            current_change_times = []
            current_first_change_times = []
            current_jump_counts = []
            current_status_cells = []
            for i in range(seq_len):
                j_idx = np.sum(jump_times[i] <= t_curr)
                tokens_so_far = history_arr[i, 0 : j_idx + 1]
                times_so_far = jump_times[i, 0 : j_idx]
                current_tokens.append(tokens_so_far[-1])
                current_change_times.append(None if j_idx == 0 else float(times_so_far[j_idx - 1]))
                first_change, last_change, jump_count = first_last_change_times(jump_times, i, t_curr)
                current_first_change_times.append(first_change)
                current_jump_counts.append(jump_count)
                current_status_cells.append(status_cell(current_tokens[-1], initial_tokens[i], jump_count))

            if args.focus_pos is not None:
                positions_to_show = [args.focus_pos]
            elif args.compact:
                positions_to_show = list(range(min(seq_len, min(4, args.max_positions_shown))))
            else:
                positions_to_show = list(range(min(seq_len, args.max_positions_shown)))

            for i in positions_to_show:
                j_idx = np.sum(jump_times[i] <= t_curr)
                tokens_so_far = history_arr[i, 0 : j_idx + 1]
                times_so_far = jump_times[i, 0 : j_idx]
                history_line = render_position_history_line(tokenizer, tokens_so_far, times_so_far, args.max_hops_shown)
                write_line(f"Pos {i:2}: {history_line}")
            if len(positions_to_show) < seq_len:
                write_line(f"{GRAY}Hidden positions: {len(positions_to_show)} shown / {seq_len} total{RESET}")

            write_line()
            write_line("-" * 115)
            summary = format_colored_sequence(tokenizer, current_tokens, current_change_times).replace("\n", " ")
            seq_label = "FINAL SEQ" if step == args.steps else "CURRENT SEQ"
            write_line(f"{BOLD}{seq_label}:{RESET}")
            write_line(
                render_sequence_cells(
                    tokenizer,
                    current_tokens,
                    current_change_times,
                    "tokens",
                    max_positions=args.max_positions_shown,
                )
            )
            write_line(f"{GRAY}decoded:{RESET} {summary}")
            current_changed = int(sum(int(curr) != int(init) for curr, init in zip(current_tokens, initial_tokens)))
            snapshot_changed = int(sum(int(snap) != int(init) for snap, init in zip(snapshot_tokens, initial_tokens)))
            snapshot_status = "reached" if t_curr >= snapshot_t else "pending"
            write_line(
                f"{GRAY}Changed now: {current_changed:>2}/{seq_len:<2} | "
                f"Snapshot: {snapshot_status:<7} | changed: {snapshot_changed:>2}/{seq_len:<2} | "
                f"t_snap={snapshot_t:0.3f}{RESET}"
            )
            if t_curr >= snapshot_t:
                write_line(f"{BOLD}SNAPSHOT t={snapshot_t:.3f}:{RESET}")
                write_line(
                    render_sequence_cells(
                        tokenizer,
                        snapshot_tokens,
                        snapshot_change_times,
                        "tokens",
                        max_positions=args.max_positions_shown,
                    )
                )
                write_line(f"{GRAY}decoded:{RESET} {snapshot_sequence}")
            write_block(
                render_token_comparison(
                    tokenizer,
                    initial_tokens,
                    current_tokens,
                    current_change_times,
                    current_first_change_times,
                    current_jump_counts,
                    current_status_cells,
                    snapshot_tokens,
                    snapshot_change_times,
                    snapshot_status_cells,
                    max_positions=args.max_positions_shown,
                )
            )
            if step == args.steps:
                write_line(
                    f"{BOLD}FINAL SUMMARY{RESET}  "
                    f"focus={args.focus_pos if args.focus_pos is not None else 'none'} | "
                    f"shown_positions={min(seq_len, args.max_positions_shown)} | "
                    f"max_hops_shown={args.max_hops_shown}"
                )
            write_line("-" * 115)

            write_line(f"{BOLD}Integrated Intensity Profile:{RESET}")
            snapshot_alpha = float(schedule.alpha_t(torch.tensor([snapshot_t], dtype=torch.float32, device=device)).item())
            snapshot_lambda = float(-torch.log(torch.tensor(snapshot_alpha, dtype=torch.float32).clamp(min=1e-10)).item())
            write_line(
                f"{GRAY}Schedule: LogLinear | current α={alpha_t:0.4f}, Λ={lambda_t:0.3f} | "
                f"snapshot α={snapshot_alpha:0.4f}, Λ={snapshot_lambda:0.3f}{RESET}"
            )
            write_block(render_lambda_plot(schedule, t_curr, snapshot_t))
            write_block(render_jump_histogram(jump_times))
            write_line(
                f"{GRAY}Markers: █=current time  {CYAN}◆{RESET}{GRAY}=snapshot time  "
                f"Colors: green->yellow->red = later semantic jumps{RESET}"
            )
            write_line(
                f"{GRAY}Snapshot coloring: unchanged tokens stay plain; changed tokens use their last-change color | "
                f"metric={args.metric:<8} beta={args.beta:0.2f}{RESET}"
            )
            write_line("-" * 115)
            sys.stdout.flush()

            time.sleep(1.0 / args.fps)

    except KeyboardInterrupt:
        print("\nAnimation stopped.")


if __name__ == "__main__":
    main()
