#!/usr/bin/env python3
"""Plot λ(t) uniform mixture schedule in the terminal."""

import os
import sys
import numpy as np
from pathlib import Path
from omegaconf import OmegaConf

# Add project root to path
repo_root = Path(__file__).resolve().parents[3]
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))

def lambda_t(t, t0, s, lam_min, t_end=1.0):
    """Normalized time-space sigmoid λ(t) from SIKForwardProcess."""
    g = _sigmoid(s * (t - t0))
    g0 = _sigmoid(s * (0.0 - t0))
    g1 = _sigmoid(s * (t_end - t0))
    g_norm = (g - g0) / (g1 - g0 + 1e-12)
    g_norm = np.clip(g_norm, 0.0, 1.0)
    return lam_min + (1.0 - lam_min) * g_norm

def plot_terminal(x, y, width=80, height=20):
    """Simple terminal scatter plot."""
    canvas = [[' ' for _ in range(width)] for _ in range(height)]
    x_min, x_max = 0.0, 1.0
    y_min, y_max = 0.0, 1.0
    
    for xi, yi in zip(x, y):
        col = int((xi - x_min) / (x_max - x_min) * (width - 1))
        row = int((yi - y_min) / (y_max - y_min) * (height - 1))
        canvas[height - 1 - row][col] = '*'

    lines = []
    lines.append(" λ(t) |" + "_" * width)
    for i, row in enumerate(canvas):
        val = 1.0 - (i / (height - 1))
        prefix = f"{val:4.1f} | "
        lines.append(prefix + "".join(row))
    lines.append("      |" + "^" * width)
    lines.append("      " + "0.0" + " " * (width // 2 - 3) + "0.5" + " " * (width // 2 - 2) + "1.0 t")
    return "\n".join(lines)

def main():
    config_path = repo_root / "configs" / "forward_process" / "sik_knn.yaml"
    if not config_path.exists():
        print(f"Error: {config_path} not found.")
        return

    cfg = OmegaConf.load(config_path)
    lam_min = cfg.get("lambda_min", 0.01)
    s = cfg.get("lambda_sigmoid_s", 5.0)
    t0 = cfg.get("lambda_t0", 0.4)
    
    print("=" * 80)
    print(f"λ(t) UNIFORM MIXTURE SCHEDULE (Terminal Plot)")
    print("=" * 80)
    print(f"Parameters from {config_path}:")
    print(f"  lambda_min:       {lam_min}")
    print(f"  lambda_sigmoid_s: {s}")
    print(f"  lambda_t0:        {t0}")
    print("-" * 80)

    t_grid = np.linspace(0.0, 1.0, 100)
    lam_grid = lambda_t(t_grid, t0, s, lam_min)
    print(plot_terminal(t_grid, lam_grid))
    print("-" * 80)
    print("DESCRIPTION: y-axis λ(t) is uniform teleport probability. x-axis is time.")
    print("=" * 80)

if __name__ == "__main__":
    main()
