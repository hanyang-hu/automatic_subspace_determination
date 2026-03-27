"""Run synthetic high-dimensional Bayesian optimization benchmarks.

This script evaluates four surrogate model variants:
  - standard
  - projected
  - linear_embedding
  - composite

Benchmarks:
  - Synthetic (BoTorch): Ackley, Rastrigin, Levy

For each synthetic benchmark and model, the script runs repeated BO with seeds 41-45 by
default, saves per-run traces to CSV (including raw observations), saves aggregate
means/std to CSV, and writes summary plots.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import gpytorch
import matplotlib.pyplot as plt
import numpy as np
import torch
from botorch.acquisition.analytic import LogExpectedImprovement
from botorch.optim import optimize_acqf
from botorch.test_functions import Ackley, Levy, Rastrigin

from manifold_optim import RiemannianAdam
from models import CompositeGPModel, LinearEmbeddingGPModel, ProjectedGPModel, StandardGPModel


MODEL_REGISTRY = {
    "standard": StandardGPModel,
    "projected": ProjectedGPModel,
    "linear_embedding": LinearEmbeddingGPModel,
    "composite": CompositeGPModel,
}


@dataclass
class BenchmarkSpec:
    name: str
    ambient_dim: int
    subspace_dim: int
    objective: Callable[[torch.Tensor], torch.Tensor]


def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)


class EmbeddedSyntheticObjective:
    def __init__(self, func, ambient_dim: int, latent_dim: int, seed: int) -> None:
        self.func = func
        self.ambient_dim = ambient_dim
        self.latent_dim = latent_dim
        gen = torch.Generator(device="cpu").manual_seed(seed)
        q, _ = torch.linalg.qr(torch.randn(ambient_dim, latent_dim, generator=gen, dtype=torch.float), mode="reduced")
        self.proj = q[:, :latent_dim]

        # BoTorch test function bounds shape: 2 x d
        self.lower = func.bounds[0].to(dtype=torch.float)
        self.upper = func.bounds[1].to(dtype=torch.float)

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        # x in [0, 1]^D -> latent coords in function domain.
        z_unit = (x - 0.5) @ self.proj
        z_unit = torch.sigmoid(2.0 * z_unit)  # map to (0, 1)
        z = self.lower + z_unit * (self.upper - self.lower)
        # test functions are minimization; maximize negative.
        val = self.func(z)
        return -val


def build_benchmarks(synth_ambient_dim: int) -> list[BenchmarkSpec]:
    synth_latent_dim = 6
    benches: list[BenchmarkSpec] = [
        BenchmarkSpec(
            name="synthetic_ackley",
            ambient_dim=synth_ambient_dim,
            subspace_dim=synth_latent_dim,
            objective=EmbeddedSyntheticObjective(Ackley(dim=synth_latent_dim), synth_ambient_dim, synth_latent_dim, seed=101),
        ),
        BenchmarkSpec(
            name="synthetic_rastrigin",
            ambient_dim=synth_ambient_dim,
            subspace_dim=synth_latent_dim,
            objective=EmbeddedSyntheticObjective(Rastrigin(dim=synth_latent_dim), synth_ambient_dim, synth_latent_dim, seed=202),
        ),
        BenchmarkSpec(
            name="synthetic_levy",
            ambient_dim=synth_ambient_dim,
            subspace_dim=synth_latent_dim,
            objective=EmbeddedSyntheticObjective(Levy(dim=synth_latent_dim), synth_ambient_dim, synth_latent_dim, seed=303),
        ),
    ]
    return benches


@dataclass
class BOConfig:
    train_steps: int
    lr: float
    acq_num_restarts: int
    acq_raw_samples: int
    stagnation_patience: int


def split_parameter_groups(model: torch.nn.Module) -> tuple[list[torch.nn.Parameter], list[torch.nn.Parameter]]:
    manifold_params: list[torch.nn.Parameter] = []
    if hasattr(model, "W") and getattr(model, "uses_riemannian_projection", False):
        manifold_params.append(model.W)

    manifold_ids = {id(p) for p in manifold_params}
    euclidean_params = [p for p in model.parameters() if p.requires_grad and id(p) not in manifold_ids]
    return manifold_params, euclidean_params


def fit_surrogate(
    model_name: str,
    train_x: torch.Tensor,
    train_y: torch.Tensor,
    subspace_dim: int,
    train_steps: int,
    lr: float,
) -> tuple[torch.nn.Module, gpytorch.likelihoods.GaussianLikelihood]:
    likelihood = gpytorch.likelihoods.GaussianLikelihood()
    model_cls = MODEL_REGISTRY[model_name]
    kwargs = dict(
        train_x=train_x,
        train_y=train_y,
        likelihood=likelihood,
        input_dim=train_x.shape[-1],
        subspace_dim=subspace_dim,
    )
    if model_name == "composite":
        kwargs["eps_alpha"] = 0.1

    model = model_cls(**kwargs)
    model.train()
    likelihood.train()

    manifold_params, euclidean_params = split_parameter_groups(model)
    euclidean_opt = torch.optim.Adam(euclidean_params, lr=lr)
    manifold_opt = RiemannianAdam(manifold_params, lr=lr) if manifold_params else None

    mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)
    for _ in range(train_steps):
        euclidean_opt.zero_grad(set_to_none=True)
        if manifold_opt is not None:
            manifold_opt.zero_grad(set_to_none=True)

        output = model(train_x)
        loss = -mll(output, train_y)
        loss.backward()

        euclidean_opt.step()
        if manifold_opt is not None:
            manifold_opt.step()

    return model.eval(), likelihood.eval()


def pick_next_point(
    model: torch.nn.Module,
    dim: int,
    best_f: float,
    num_restarts: int,
    raw_samples: int,
) -> torch.Tensor:
    bounds = torch.stack(
        [torch.zeros(dim, dtype=torch.float), torch.ones(dim, dtype=torch.float)]
    )
    acqf = LogExpectedImprovement(model=model, best_f=best_f)
    candidates, _ = optimize_acqf(
        acq_function=acqf,
        bounds=bounds,
        q=1,
        num_restarts=num_restarts,
        raw_samples=raw_samples,
    )
    return candidates


def run_single_bo(
    benchmark: BenchmarkSpec,
    model_name: str,
    seed: int,
    n_init: int,
    n_iter: int,
    config: BOConfig,
    print_every: int,
) -> list[dict]:
    set_seed(seed)

    sobol = torch.quasirandom.SobolEngine(dimension=benchmark.ambient_dim, scramble=True, seed=seed)
    train_x = sobol.draw(n_init).to(dtype=torch.float)
    train_y = benchmark.objective(train_x).to(dtype=torch.float)

    incumbent = float(torch.max(train_y).item())
    rows: list[dict] = [{"iteration": 0, "observed_value": incumbent, "best_observed": incumbent}]
    no_improvement_steps = 0
    for bo_iter in range(1, n_iter + 1):
        y_mean = train_y.mean()
        y_std = train_y.std(unbiased=False).clamp_min(1e-6)
        train_y_norm = (train_y - y_mean) / y_std

        model, _ = fit_surrogate(
            model_name=model_name,
            train_x=train_x,
            train_y=train_y_norm,
            subspace_dim=benchmark.subspace_dim,
            train_steps=config.train_steps,
            lr=config.lr,
        )

        if config.stagnation_patience > 0 and no_improvement_steps >= config.stagnation_patience:
            x_next = torch.rand(1, benchmark.ambient_dim, dtype=torch.float)
            no_improvement_steps = 0
        else:
            x_next = pick_next_point(
                model=model,
                dim=benchmark.ambient_dim,
                best_f=float(train_y_norm.max().item()),
                num_restarts=config.acq_num_restarts,
                raw_samples=config.acq_raw_samples,
            )
        y_next = benchmark.objective(x_next).to(dtype=torch.float)
        observed_value = float(y_next.item())

        train_x = torch.cat([train_x, x_next], dim=0)
        train_y = torch.cat([train_y, y_next], dim=0)
        prev_incumbent = incumbent
        incumbent = max(incumbent, observed_value)
        improved = observed_value > prev_incumbent + 1e-12
        no_improvement_steps = 0 if improved else no_improvement_steps + 1

        rows.append({"iteration": bo_iter, "observed_value": observed_value, "best_observed": incumbent})
        if print_every > 0 and bo_iter % print_every == 0:
            print(
                f"[ITER {bo_iter:04d}] observed={observed_value:.6f} "
                f"incumbent={incumbent:.6f} stalled_for={no_improvement_steps}"
            )

    return rows


def save_run_trace(run_rows: list[dict], out_dir: Path, seed: int) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    run_csv = out_dir / f"bo_run_seed_{seed}.csv"
    with run_csv.open("w", encoding="utf-8") as f:
        f.write("benchmark,model,seed,iteration,observed_value,best_observed\n")
        for row in run_rows:
            f.write(
                f"{row['benchmark']},{row['model']},{row['seed']},{row['iteration']},"
                f"{row['observed_value']:.10f},{row['best_observed']:.10f}\n"
            )
    return run_csv



def save_run_plot(run_rows: list[dict], out_dir: Path, seed: int) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    x = np.array([row["iteration"] for row in run_rows], dtype=float)
    y = np.array([row["best_observed"] for row in run_rows], dtype=float)

    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(8, 4))
    ax.plot(x, y, linewidth=1.5)
    ax.set_title(f"{run_rows[0]['benchmark']} | {run_rows[0]['model']} | seed={seed}")
    ax.set_xlabel("BO iteration")
    ax.set_ylabel("Best observed value")
    ax.grid(alpha=0.2)

    fig.tight_layout()
    plot_path = out_dir / f"bo_run_seed_{seed}.png"
    fig.savefig(plot_path, dpi=180)
    plt.close(fig)
    return plot_path

def aggregate_and_save(all_rows: list[dict], out_dir: Path) -> list[dict]:
    out_dir.mkdir(parents=True, exist_ok=True)

    all_csv = out_dir / "bo_results_all_runs.csv"
    with all_csv.open("w", encoding="utf-8") as f:
        f.write("benchmark,model,seed,iteration,observed_value,best_observed\n")
        for row in all_rows:
            f.write(
                f"{row['benchmark']},{row['model']},{row['seed']},{row['iteration']},"
                f"{row['observed_value']:.10f},{row['best_observed']:.10f}\n"
            )

    grouped: dict[tuple[str, str, int], list[float]] = {}
    for row in all_rows:
        key = (row["benchmark"], row["model"], row["iteration"])
        grouped.setdefault(key, []).append(row["best_observed"])

    agg_rows: list[dict] = []
    for (benchmark, model, iteration), vals in sorted(grouped.items()):
        arr = np.array(vals, dtype=float)
        agg_rows.append(
            {
                "benchmark": benchmark,
                "model": model,
                "iteration": int(iteration),
                "best_observed_mean": float(np.mean(arr)),
                "best_observed_std": float(np.std(arr, ddof=1)) if len(arr) > 1 else 0.0,
            }
        )

    agg_csv = out_dir / "bo_results_aggregated.csv"
    with agg_csv.open("w", encoding="utf-8") as f:
        f.write("benchmark,model,iteration,best_observed_mean,best_observed_std\n")
        for row in agg_rows:
            f.write(
                f"{row['benchmark']},{row['model']},{row['iteration']},{row['best_observed_mean']:.10f},{row['best_observed_std']:.10f}\n"
            )

    return agg_rows


def plot_aggregated(agg_rows: list[dict], out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    if not agg_rows:
        raise ValueError("No rows to plot.")

    benchmark_name = agg_rows[0]["benchmark"]
    model_name = agg_rows[0]["model"]
    rows = sorted(agg_rows, key=lambda r: r["iteration"])
    x = np.array([r["iteration"] for r in rows], dtype=float)
    m = np.array([r["best_observed_mean"] for r in rows], dtype=float)
    s = np.array([r["best_observed_std"] for r in rows], dtype=float)

    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(9, 4.5))
    ax.plot(x, m, label=model_name)
    ax.fill_between(x, m - s, m + s, alpha=0.15)

    ax.set_title(f"{benchmark_name} | {model_name}")
    ax.set_xlabel("BO iteration")
    ax.set_ylabel("Best observed value")
    ax.grid(alpha=0.2)
    ax.legend(loc="best")

    fig.tight_layout()
    plot_path = out_dir / "bo_results_plot.png"
    fig.savefig(plot_path, dpi=180)
    plt.close(fig)
    return plot_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output_dir", type=Path, default=Path("outputs"))
    parser.add_argument("--synthetic_ambient_dim", type=int, default=50)
    parser.add_argument("--n_init", type=int, default=50)
    parser.add_argument("--n_iter", type=int, default=100)
    parser.add_argument("--train_steps", type=int, default=50)
    parser.add_argument("--lr", type=float, default=0.05)
    parser.add_argument("--acq_num_restarts", type=int, default=5)
    parser.add_argument("--acq_raw_samples", type=int, default=256)
    parser.add_argument("--stagnation_patience", type=int, default=20)
    parser.add_argument("--standard_train_steps_mult", type=float, default=2.0)
    parser.add_argument("--standard_lr_mult", type=float, default=0.5)
    parser.add_argument("--standard_acq_mult", type=float, default=2.0)
    parser.add_argument("--print_every", type=int, default=10)
    parser.add_argument("--seeds", type=int, nargs="+", default=[41, 42, 43, 44, 45])
    parser.add_argument("--models", nargs="+", default=list(MODEL_REGISTRY.keys()), choices=list(MODEL_REGISTRY.keys()))
    return parser.parse_args()


def build_config(args: argparse.Namespace, model_name: str) -> BOConfig:
    train_steps = args.train_steps
    lr = args.lr
    acq_num_restarts = args.acq_num_restarts
    acq_raw_samples = args.acq_raw_samples

    if model_name == "standard":
        train_steps = max(1, int(round(train_steps * args.standard_train_steps_mult)))
        lr = max(1e-4, lr * args.standard_lr_mult)
        acq_num_restarts = max(1, int(round(acq_num_restarts * args.standard_acq_mult)))
        acq_raw_samples = max(32, int(round(acq_raw_samples * args.standard_acq_mult)))

    return BOConfig(
        train_steps=train_steps,
        lr=lr,
        acq_num_restarts=acq_num_restarts,
        acq_raw_samples=acq_raw_samples,
        stagnation_patience=args.stagnation_patience,
    )


def main() -> None:
    args = parse_args()
    benchmarks = build_benchmarks(synth_ambient_dim=args.synthetic_ambient_dim)

    for bench in benchmarks:
        for model_name in args.models:
            config = build_config(args, model_name)
            method_dir = args.output_dir / bench.name / model_name
            combo_rows: list[dict] = []
            for seed in args.seeds:
                print(f"[RUN] benchmark={bench.name} model={model_name} seed={seed}")
                print(
                    "[CONFIG] "
                    f"train_steps={config.train_steps} lr={config.lr:.6f} "
                    f"acq_num_restarts={config.acq_num_restarts} acq_raw_samples={config.acq_raw_samples} "
                    f"stagnation_patience={config.stagnation_patience}"
                )
                trace_rows = run_single_bo(
                    benchmark=bench,
                    model_name=model_name,
                    seed=seed,
                    n_init=args.n_init,
                    n_iter=args.n_iter,
                    config=config,
                    print_every=args.print_every,
                )
                run_rows: list[dict] = []
                for row_data in trace_rows:
                    row = {
                        "benchmark": bench.name,
                        "model": model_name,
                        "seed": seed,
                        "iteration": row_data["iteration"],
                        "observed_value": row_data["observed_value"],
                        "best_observed": row_data["best_observed"],
                    }
                    run_rows.append(row)
                    combo_rows.append(row)

                run_csv_path = save_run_trace(run_rows, method_dir, seed=seed)
                run_plot_path = save_run_plot(run_rows, method_dir, seed=seed)
                print(f"Saved run trace to: {run_csv_path}")
                print(f"Saved run plot to: {run_plot_path}")

            agg_csv_rows = aggregate_and_save(combo_rows, method_dir)
            plot_path = plot_aggregated(agg_csv_rows, method_dir)
            print(f"Saved aggregate results to: {method_dir / 'bo_results_aggregated.csv'}")
            print(f"Saved plot to: {plot_path}")


if __name__ == "__main__":
    main()
