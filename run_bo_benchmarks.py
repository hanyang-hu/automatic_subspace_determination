"""Run high-dimensional Bayesian optimization benchmarks across four GP models.

This script evaluates four surrogate model variants:
  - standard
  - projected
  - linear_embedding
  - composite

Benchmarks:
  - Synthetic (BoTorch): Ackley, Rastrigin, Levy
  - Real-world (sklearn): LASSO CV, RBF-SVM CV

For each benchmark and model, the script runs repeated BO with seeds 41-45 by default,
saves per-run traces to CSV, saves aggregate means/std to CSV, and writes summary plots.
"""

from __future__ import annotations

import argparse
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import gpytorch
import matplotlib.pyplot as plt
import numpy as np
import torch
from botorch.test_functions import Ackley, Levy, Rastrigin
from sklearn.datasets import load_breast_cancer, load_diabetes
from sklearn.linear_model import Lasso
from sklearn.model_selection import KFold, cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

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
        q, _ = torch.linalg.qr(torch.randn(ambient_dim, latent_dim, generator=gen, dtype=torch.double), mode="reduced")
        self.proj = q[:, :latent_dim]

        # BoTorch test function bounds shape: 2 x d
        self.lower = func.bounds[0].to(dtype=torch.double)
        self.upper = func.bounds[1].to(dtype=torch.double)

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        # x in [0, 1]^D -> latent coords in function domain.
        z_unit = (x - 0.5) @ self.proj
        z_unit = torch.sigmoid(2.0 * z_unit)  # map to (0, 1)
        z = self.lower + z_unit * (self.upper - self.lower)
        # test functions are minimization; maximize negative.
        val = self.func(z)
        return -val


class EmbeddedRealObjective:
    def __init__(
        self,
        ambient_dim: int,
        latent_dim: int,
        seed: int,
        task: str,
    ) -> None:
        self.ambient_dim = ambient_dim
        self.latent_dim = latent_dim
        self.task = task

        gen = torch.Generator(device="cpu").manual_seed(seed)
        q, _ = torch.linalg.qr(torch.randn(ambient_dim, latent_dim, generator=gen, dtype=torch.double), mode="reduced")
        self.proj = q[:, :latent_dim]

        if task == "lasso":
            X, y = load_diabetes(return_X_y=True)
            self.X = X
            self.y = y
            self.cv = KFold(n_splits=3, shuffle=True, random_state=seed)
        elif task == "svm":
            data = load_breast_cancer()
            self.X = data.data
            self.y = data.target
            self.cv = KFold(n_splits=3, shuffle=True, random_state=seed)
        else:
            raise ValueError(f"Unsupported real objective task: {task}")

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        # x in [0,1]^D -> u in R^latent_dim
        u = (x - 0.5) @ self.proj
        u = torch.sigmoid(2.5 * u)

        vals: list[float] = []
        for row in u:
            a = float(row[0].detach().cpu())
            b = float(row[1].detach().cpu()) if self.latent_dim > 1 else 0.5

            if self.task == "lasso":
                alpha = 10 ** (-4 + 4 * a)  # [1e-4, 1]
                model = make_pipeline(StandardScaler(), Lasso(alpha=alpha, max_iter=5000, random_state=0))
                score = cross_val_score(
                    model,
                    self.X,
                    self.y,
                    cv=self.cv,
                    scoring="neg_mean_squared_error",
                    n_jobs=1,
                ).mean()
            else:  # svm
                C = 10 ** (-2 + 4 * a)  # [1e-2, 1e2]
                gamma = 10 ** (-4 + 4 * b)  # [1e-4, 1]
                model = make_pipeline(StandardScaler(), SVC(C=C, gamma=gamma, kernel="rbf", random_state=0))
                score = cross_val_score(
                    model,
                    self.X,
                    self.y,
                    cv=self.cv,
                    scoring="accuracy",
                    n_jobs=1,
                ).mean()
            vals.append(float(score))

        return torch.tensor(vals, dtype=torch.double)


def build_benchmarks(ambient_dim: int) -> list[BenchmarkSpec]:
    synth_latent_dim = 6
    real_latent_dim = 2
    benches: list[BenchmarkSpec] = [
        BenchmarkSpec(
            name="synthetic_ackley",
            ambient_dim=ambient_dim,
            subspace_dim=synth_latent_dim,
            objective=EmbeddedSyntheticObjective(Ackley(dim=synth_latent_dim), ambient_dim, synth_latent_dim, seed=101),
        ),
        BenchmarkSpec(
            name="synthetic_rastrigin",
            ambient_dim=ambient_dim,
            subspace_dim=synth_latent_dim,
            objective=EmbeddedSyntheticObjective(Rastrigin(dim=synth_latent_dim), ambient_dim, synth_latent_dim, seed=202),
        ),
        BenchmarkSpec(
            name="synthetic_levy",
            ambient_dim=ambient_dim,
            subspace_dim=synth_latent_dim,
            objective=EmbeddedSyntheticObjective(Levy(dim=synth_latent_dim), ambient_dim, synth_latent_dim, seed=303),
        ),
        BenchmarkSpec(
            name="real_lasso_cv",
            ambient_dim=ambient_dim,
            subspace_dim=real_latent_dim,
            objective=EmbeddedRealObjective(ambient_dim, real_latent_dim, seed=404, task="lasso"),
        ),
        BenchmarkSpec(
            name="real_svm_cv",
            ambient_dim=ambient_dim,
            subspace_dim=real_latent_dim,
            objective=EmbeddedRealObjective(ambient_dim, real_latent_dim, seed=505, task="svm"),
        ),
    ]
    return benches


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


@torch.no_grad()
def pick_next_point(
    model: torch.nn.Module,
    likelihood: gpytorch.likelihoods.GaussianLikelihood,
    dim: int,
    num_candidates: int,
    beta: float,
) -> torch.Tensor:
    sobol = torch.quasirandom.SobolEngine(dimension=dim, scramble=True)
    cand = sobol.draw(num_candidates).to(dtype=torch.double)

    pred = likelihood(model(cand))
    mean = pred.mean
    std = pred.variance.clamp_min(1e-12).sqrt()
    ucb = mean + beta * std
    idx = int(torch.argmax(ucb).item())
    return cand[idx : idx + 1]


def run_single_bo(
    benchmark: BenchmarkSpec,
    model_name: str,
    seed: int,
    n_init: int,
    n_iter: int,
    train_steps: int,
    lr: float,
    num_candidates: int,
) -> list[float]:
    set_seed(seed)

    sobol = torch.quasirandom.SobolEngine(dimension=benchmark.ambient_dim, scramble=True, seed=seed)
    train_x = sobol.draw(n_init).to(dtype=torch.double)
    train_y = benchmark.objective(train_x).to(dtype=torch.double)

    best_trace = [float(torch.max(train_y).item())]
    for _ in range(n_iter):
        y_mean = train_y.mean()
        y_std = train_y.std(unbiased=False).clamp_min(1e-6)
        train_y_norm = (train_y - y_mean) / y_std

        model, likelihood = fit_surrogate(
            model_name=model_name,
            train_x=train_x,
            train_y=train_y_norm,
            subspace_dim=benchmark.subspace_dim,
            train_steps=train_steps,
            lr=lr,
        )

        beta = 0.2 * math.log(len(train_x) + 1)
        x_next = pick_next_point(
            model=model,
            likelihood=likelihood,
            dim=benchmark.ambient_dim,
            num_candidates=num_candidates,
            beta=beta,
        )
        y_next = benchmark.objective(x_next).to(dtype=torch.double)

        train_x = torch.cat([train_x, x_next], dim=0)
        train_y = torch.cat([train_y, y_next], dim=0)
        best_trace.append(float(torch.max(train_y).item()))

    return best_trace


def aggregate_and_save(all_rows: list[dict], out_dir: Path) -> list[dict]:
    out_dir.mkdir(parents=True, exist_ok=True)

    all_csv = out_dir / "bo_results_all_runs.csv"
    with all_csv.open("w", encoding="utf-8") as f:
        f.write("benchmark,model,seed,iteration,best_observed\n")
        for row in all_rows:
            f.write(
                f"{row['benchmark']},{row['model']},{row['seed']},{row['iteration']},{row['best_observed']:.10f}\n"
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


def plot_aggregated(agg_rows: list[dict], out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    benchmarks = sorted({r["benchmark"] for r in agg_rows})
    n = len(benchmarks)
    fig, axes = plt.subplots(nrows=n, ncols=1, figsize=(9, 3.5 * n), sharex=False)
    if n == 1:
        axes = [axes]

    for ax, bench in zip(axes, benchmarks):
        bench_rows = [r for r in agg_rows if r["benchmark"] == bench]
        model_names = sorted({r["model"] for r in bench_rows})
        for model_name in model_names:
            rows = sorted(
                (r for r in bench_rows if r["model"] == model_name),
                key=lambda r: r["iteration"],
            )
            x = np.array([r["iteration"] for r in rows], dtype=float)
            m = np.array([r["best_observed_mean"] for r in rows], dtype=float)
            s = np.array([r["best_observed_std"] for r in rows], dtype=float)
            ax.plot(x, m, label=model_name)
            ax.fill_between(x, m - s, m + s, alpha=0.15)

        ax.set_title(bench)
        ax.set_xlabel("BO iteration")
        ax.set_ylabel("Best observed value")
        ax.grid(alpha=0.2)
        ax.legend(loc="best")

    fig.tight_layout()
    fig.savefig(out_dir / "bo_results_plot.png", dpi=180)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results_dir", type=Path, default=Path("results/bo"))
    parser.add_argument("--ambient_dim", type=int, default=20)
    parser.add_argument("--n_init", type=int, default=12)
    parser.add_argument("--n_iter", type=int, default=25)
    parser.add_argument("--train_steps", type=int, default=80)
    parser.add_argument("--lr", type=float, default=0.05)
    parser.add_argument("--num_candidates", type=int, default=1024)
    parser.add_argument("--seeds", type=int, nargs="+", default=[41, 42, 43, 44, 45])
    parser.add_argument("--models", nargs="+", default=list(MODEL_REGISTRY.keys()), choices=list(MODEL_REGISTRY.keys()))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    benchmarks = build_benchmarks(ambient_dim=args.ambient_dim)

    all_rows: list[dict] = []
    for bench in benchmarks:
        for model_name in args.models:
            for seed in args.seeds:
                print(f"[RUN] benchmark={bench.name} model={model_name} seed={seed}")
                trace = run_single_bo(
                    benchmark=bench,
                    model_name=model_name,
                    seed=seed,
                    n_init=args.n_init,
                    n_iter=args.n_iter,
                    train_steps=args.train_steps,
                    lr=args.lr,
                    num_candidates=args.num_candidates,
                )
                for it, best in enumerate(trace):
                    all_rows.append(
                        {
                            "benchmark": bench.name,
                            "model": model_name,
                            "seed": seed,
                            "iteration": it,
                            "best_observed": best,
                        }
                    )

    agg = aggregate_and_save(all_rows, args.results_dir)
    plot_aggregated(agg, args.results_dir)
    print(f"Saved run-level results to: {args.results_dir / 'bo_results_all_runs.csv'}")
    print(f"Saved aggregate results to: {args.results_dir / 'bo_results_aggregated.csv'}")
    print(f"Saved plot to: {args.results_dir / 'bo_results_plot.png'}")


if __name__ == "__main__":
    main()
