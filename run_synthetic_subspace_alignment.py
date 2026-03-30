"""Measure learned subspace alignment versus dataset size on embedded Ackley.

Setup:
- 6D Ackley latent objective embedded in 50D ambient space (default).
- Data sampled from Sobol sequence in [0, 1]^D.
- At each step we add more data, warm-starting model parameters from previous step.
- Repeat over multiple random seeds, then save mean/std alignment curves.

The primary alignment metric is Grassmann geodesic distance between the true
subspace basis and the estimated basis.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from botorch.test_functions import Ackley

from run_synthetic_bo_benchmark import (
    BOConfig,
    EmbeddedSyntheticObjective,
    build_warm_start,
    fit_surrogate,
    set_seed,
)

DEFAULT_DTYPE = torch.double


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output_dir", type=Path, default=Path("outputs/subspace_alignment_ackley"))
    parser.add_argument("--ambient_dim", type=int, default=50)
    parser.add_argument("--latent_dim", type=int, default=6)
    parser.add_argument(
        "--models",
        nargs="+",
        default=["projected", "linear_embedding", "composite"],
        choices=["projected", "linear_embedding", "composite"],
        help="Models that learn a subspace matrix W.",
    )
    parser.add_argument("--n_start", type=int, default=50)
    parser.add_argument("--n_step", type=int, default=25, help="New Sobol points added each step.")
    parser.add_argument("--n_steps", type=int, default=8, help="Number of growth steps after n_start.")
    parser.add_argument("--num_repeats", type=int, default=5)
    parser.add_argument("--seeds", type=int, nargs="+", default=[41, 42, 43, 44, 45])

    parser.add_argument("--subspace_dim", type=int, default=6, help="Learned model subspace dimension.")
    parser.add_argument("--objective_seed", type=int, default=101)

    parser.add_argument("--train_steps", type=int, default=120)
    parser.add_argument("--lr", type=float, default=0.02)
    parser.add_argument("--manifold_lr_mult", type=float, default=1.0)
    parser.add_argument("--embedding_grad_clip", type=float, default=10.0)
    parser.add_argument("--alt_num_outer_steps", type=int, default=0)
    parser.add_argument("--alt_euclidean_steps", type=int, default=1)
    parser.add_argument("--alt_embedding_steps", type=int, default=1)
    parser.add_argument(
        "--warm_start_mode",
        type=str,
        default="w_only",
        choices=["w_only", "all"],
        help="Warm-start policy as dataset grows.",
    )
    parser.add_argument("--print_every", type=int, default=1)
    return parser.parse_args()


def _orthonormalize_cols(x: torch.Tensor) -> torch.Tensor:
    q, _ = torch.linalg.qr(x, mode="reduced")
    return q


def grassmann_geodesic_distance(w_est: torch.Tensor, w_true: torch.Tensor) -> float:
    """Grassmann geodesic distance using principal angles via eigendecomposition.

    For equal-dimensional subspaces, principal angles satisfy:
      cos^2(theta_i) = eigvals((Qe^T Qt)(Qt^T Qe))
    where Qe, Qt are orthonormal bases.
    """

    if w_est.shape != w_true.shape:
        raise ValueError(f"Shape mismatch: estimated {w_est.shape}, true {w_true.shape}")

    qe = _orthonormalize_cols(w_est)
    qt = _orthonormalize_cols(w_true)
    overlap = qe.transpose(-2, -1) @ qt
    gram = overlap.transpose(-2, -1) @ overlap
    eigvals = torch.linalg.eigvalsh(gram)
    eigvals = torch.clamp(eigvals, 0.0, 1.0)
    principal_angles = torch.acos(torch.sqrt(eigvals))
    return float(torch.linalg.norm(principal_angles, ord=2).detach().cpu())


def write_rows_csv(rows: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "model",
                "seed",
                "repeat",
                "n_train",
                "geodesic_distance",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)


def aggregate_rows(rows: list[dict]) -> list[dict]:
    grouped: dict[tuple[str, int], list[float]] = {}
    for row in rows:
        key = (row["model"], row["n_train"])
        grouped.setdefault(key, []).append(row["geodesic_distance"])

    out: list[dict] = []
    for (model, n_train), vals in sorted(grouped.items(), key=lambda x: (x[0][0], x[0][1])):
        arr = np.array(vals, dtype=float)
        out.append(
            {
                "model": model,
                "n_train": int(n_train),
                "geodesic_mean": float(np.mean(arr)),
                "geodesic_std": float(np.std(arr, ddof=1)) if len(arr) > 1 else 0.0,
            }
        )
    return out


def write_aggregated_csv(rows: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["model", "n_train", "geodesic_mean", "geodesic_std"],
        )
        writer.writeheader()
        writer.writerows(rows)


def plot_aggregated(agg_rows: list[dict], out_path: Path) -> None:
    if not agg_rows:
        raise ValueError("No aggregated rows to plot.")

    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(9, 4.8))
    models = sorted({r["model"] for r in agg_rows})
    for model in models:
        rows = [r for r in agg_rows if r["model"] == model]
        rows = sorted(rows, key=lambda r: r["n_train"])
        x = np.array([r["n_train"] for r in rows], dtype=float)
        mean = np.array([r["geodesic_mean"] for r in rows], dtype=float)
        std = np.array([r["geodesic_std"] for r in rows], dtype=float)
        ax.plot(x, mean, marker="o", label=model)
        ax.fill_between(x, mean - std, mean + std, alpha=0.2)

    ax.set_title("Subspace alignment vs dataset size (Ackley 6D in 50D)")
    ax.set_xlabel("Number of training points")
    ax.set_ylabel("Grassmann geodesic distance (lower is better)")
    ax.grid(alpha=0.2)
    ax.legend(loc="best")

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    if len(args.seeds) < args.num_repeats:
        raise ValueError("Please provide at least --num_repeats seeds.")
    if args.subspace_dim != args.latent_dim:
        raise ValueError(
            "Geodesic distance requires matching dimensions. "
            "Please set --subspace_dim equal to --latent_dim."
        )

    config = BOConfig(
        train_steps=args.train_steps,
        lr=args.lr,
        manifold_lr_mult=args.manifold_lr_mult,
        embedding_grad_clip=args.embedding_grad_clip,
        alt_num_outer_steps=args.alt_num_outer_steps,
        alt_euclidean_steps=args.alt_euclidean_steps,
        alt_embedding_steps=args.alt_embedding_steps,
        acq_num_restarts=8,
        acq_raw_samples=64,
        stagnation_patience=0,
        acq_opt_space="ambient",
    )

    objective = EmbeddedSyntheticObjective(
        Ackley(dim=args.latent_dim),
        ambient_dim=args.ambient_dim,
        latent_dim=args.latent_dim,
        seed=args.objective_seed,
    )
    w_true = objective.proj

    step_sizes = [args.n_start + i * args.n_step for i in range(args.n_steps + 1)]
    n_max = max(step_sizes)
    all_rows: list[dict] = []

    for model_name in args.models:
        for repeat_idx, seed in enumerate(args.seeds[: args.num_repeats]):
            set_seed(seed)
            sobol = torch.quasirandom.SobolEngine(
                dimension=args.ambient_dim,
                scramble=True,
                seed=seed,
            )
            full_x = sobol.draw(n_max).to(dtype=DEFAULT_DTYPE)
            full_y = objective(full_x).to(dtype=DEFAULT_DTYPE)

            warm_start = None
            for step_idx, n_train in enumerate(step_sizes):
                train_x = full_x[:n_train]
                train_y = full_y[:n_train]

                y_mean = train_y.mean()
                y_std = train_y.std(unbiased=False).clamp_min(1e-6)
                train_y_norm = (train_y - y_mean) / y_std

                model, likelihood = fit_surrogate(
                    model_name=model_name,
                    train_x=train_x,
                    train_y=train_y_norm,
                    subspace_dim=args.subspace_dim,
                    train_steps=config.train_steps,
                    lr=config.lr,
                    manifold_lr_mult=config.manifold_lr_mult,
                    embedding_grad_clip=config.embedding_grad_clip,
                    alt_num_outer_steps=config.alt_num_outer_steps,
                    alt_euclidean_steps=config.alt_euclidean_steps,
                    alt_embedding_steps=config.alt_embedding_steps,
                    warm_start=warm_start,
                    warm_start_w_only=args.warm_start_mode == "w_only",
                )
                warm_start = build_warm_start(
                    model=model,
                    likelihood=likelihood,
                    include_non_w=args.warm_start_mode == "all",
                )

                geodesic = grassmann_geodesic_distance(model.W.detach(), w_true)
                row = {
                    "model": model_name,
                    "seed": seed,
                    "repeat": repeat_idx,
                    "n_train": n_train,
                    "geodesic_distance": geodesic,
                }
                all_rows.append(row)

                if args.print_every > 0 and step_idx % args.print_every == 0:
                    print(
                        f"[MODEL={model_name}] repeat={repeat_idx} seed={seed} "
                        f"n_train={n_train} geodesic={geodesic:.6f}"
                    )

    raw_csv = args.output_dir / "alignment_all_runs.csv"
    agg_csv = args.output_dir / "alignment_aggregated.csv"
    plot_path = args.output_dir / "alignment_vs_dataset_size.png"

    write_rows_csv(all_rows, raw_csv)
    agg_rows = aggregate_rows(all_rows)
    write_aggregated_csv(agg_rows, agg_csv)
    plot_aggregated(agg_rows, plot_path)

    print(f"Saved per-run results to: {raw_csv}")
    print(f"Saved aggregated results to: {agg_csv}")
    print(f"Saved plot to: {plot_path}")


if __name__ == "__main__":
    main()
