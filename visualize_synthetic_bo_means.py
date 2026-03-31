"""Visualize mean BO traces across surrogate models for one synthetic benchmark folder.

Expected input structure (from run_synthetic_bo_benchmark.py):
  <benchmark_output_dir>/<model>/bo_results_aggregated.csv

Example:
  python visualize_synthetic_bo_means.py \
      --benchmark_output_dir outputs/synthetic_levy
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--benchmark_output_dir",
        type=Path,
        required=True,
        help="Folder containing per-model subfolders with bo_results_aggregated.csv",
    )
    parser.add_argument(
        "--output_name",
        type=str,
        default="bo_model_mean_comparison.png",
        help="Output filename saved under --benchmark_output_dir.",
    )
    return parser.parse_args()


def load_aggregated_csv(path: Path) -> tuple[np.ndarray, np.ndarray]:
    iterations: list[float] = []
    means: list[float] = []
    with path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            iterations.append(float(row["iteration"]))
            means.append(float(row["best_observed_mean"]))
    if not iterations:
        raise ValueError(f"No rows found in {path}")
    return np.array(iterations, dtype=float), np.array(means, dtype=float)


def main() -> None:
    args = parse_args()
    benchmark_dir = args.benchmark_output_dir
    if not benchmark_dir.exists():
        raise FileNotFoundError(f"Benchmark output folder does not exist: {benchmark_dir}")

    model_dirs = sorted([d for d in benchmark_dir.iterdir() if d.is_dir()])
    if not model_dirs:
        raise ValueError(f"No model subdirectories found under: {benchmark_dir}")

    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(9, 9))
    loaded_any = False

    for model_dir in model_dirs:
        agg_csv = model_dir / "bo_results_aggregated.csv"
        if not agg_csv.exists():
            print(f"[SKIP] Missing aggregated CSV for {model_dir.name}: {agg_csv}")
            continue
        x, y = load_aggregated_csv(agg_csv)
        ax.plot(x, y, linewidth=2.0, label=model_dir.name)
        loaded_any = True

    if not loaded_any:
        raise ValueError(f"No bo_results_aggregated.csv files found under: {benchmark_dir}")

    benchmark_name = {
        "synthetic_levy": "Levy 6D in 50D",
        "synthetic_ackley": "Ackley 6D in 50D",
        "synthetic_rastrigin": "Rastrigin 6D in 50D"
    }
    ax.set_title(f"{benchmark_name.get(benchmark_dir.name, benchmark_dir.name)}")
    ax.set_xlabel("BO iteration")
    ax.set_ylabel("Best Observed Value")
    ax.grid(alpha=0.2)
    ax.legend(loc="best")

    fig.tight_layout()
    out_path = benchmark_dir / args.output_name
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    print(f"Saved plot to: {out_path}")


if __name__ == "__main__":
    main()
