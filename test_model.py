"""CLI script to train and visualize GP subspace models on synthetic data."""

from __future__ import annotations

import argparse
from pathlib import Path

import gpytorch
import matplotlib.pyplot as plt
import numpy as np
import torch
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

from manifold_optim import RiemannianAdam
from models import (
    CompositeGPModel,
    LinearEmbeddingGPModel,
    ProjectedGPModel,
    StandardGPModel,
)
from utils import make_latent_function, make_linear_subspace_train_test_split, orthogonality_error, set_seed


MODEL_REGISTRY = {
    "standard": StandardGPModel,
    "projected": ProjectedGPModel,
    "linear_embedding": LinearEmbeddingGPModel,
    "composite": CompositeGPModel,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=MODEL_REGISTRY, default="composite")
    parser.add_argument("--D", type=int, default=2, help="Ambient input dimension")
    parser.add_argument("--k_true", type=int, default=1, help="True latent subspace dim")
    parser.add_argument("--k_model", type=int, default=1, help="Model latent subspace dim")
    parser.add_argument("--n_train", type=int, default=200)
    parser.add_argument("--n_test", type=int, default=200)
    parser.add_argument("--noise_std", type=float, default=0.1)
    parser.add_argument("--iters", type=int, default=300)
    parser.add_argument("--lr", type=float, default=5e-2)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    parser.add_argument("--latent", default="smooth", choices=["linear", "smooth", "nonlinear"])
    parser.add_argument("--output_dir", type=Path, default=Path("artifacts"))
    parser.add_argument("--eps_alpha", type=float, default=0.01, help="Half-Cauchy prior scale for composite eps")
    args = parser.parse_args()
    if args.D != 2:
        raise ValueError("This visualization script expects a 2D input space. Please run with --D 2.")
    if args.k_true != 1:
        raise ValueError("Toy data generation is configured for a 1D true response subspace. Please run with --k_true 1.")
    return args


def split_parameter_groups(model: torch.nn.Module) -> tuple[list[torch.nn.Parameter], list[torch.nn.Parameter]]:
    manifold_params: list[torch.nn.Parameter] = []
    if hasattr(model, "W") and getattr(model, "uses_riemannian_projection", False):
        manifold_params.append(model.W)

    manifold_ids = {id(p) for p in manifold_params}
    euclidean_params = [p for p in model.parameters() if p.requires_grad and id(p) not in manifold_ids]
    return manifold_params, euclidean_params


def train_model(model, likelihood, train_x, train_y, iters: int, lr: float):
    model.train()
    likelihood.train()

    manifold_params, euclidean_params = split_parameter_groups(model)
    euclidean_optim = torch.optim.Adam(euclidean_params, lr=lr)
    manifold_optim = RiemannianAdam(manifold_params, lr=lr) if manifold_params else None

    mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)
    losses = []

    for i in range(iters):
        euclidean_optim.zero_grad(set_to_none=True)
        if manifold_optim is not None:
            manifold_optim.zero_grad(set_to_none=True)

        output = model(train_x)
        loss = -mll(output, train_y)
        loss.backward()

        euclidean_optim.step()
        if manifold_optim is not None:
            manifold_optim.step()

        losses.append(float(loss.detach().cpu()))

        if (i + 1) % max(1, iters // 10) == 0:
            status = f"iter={i+1:04d}/{iters} loss={losses[-1]:.4f}"
            if manifold_params:
                status += f" ortho_err={orthogonality_error(manifold_params[0]).item():.2e}"
            print(status)

    return losses


@torch.no_grad()
def evaluate_model(model, likelihood, test_x, test_y):
    model.eval()
    likelihood.eval()

    with gpytorch.settings.fast_pred_var():
        pred_dist = likelihood(model(test_x))

    mean = pred_dist.mean
    rmse = torch.sqrt(torch.mean((mean - test_y) ** 2)).item()
    nll = -pred_dist.log_prob(test_y).item() / test_y.numel()
    return rmse, nll, mean, pred_dist


@torch.no_grad()
def visualize_results(
    output_dir: Path,
    model_name: str,
    model,
    train_x,
    train_y_denorm,
    test_x,
    test_y_denorm,
    pred_mean_denorm,
    losses,
    W_true,
    latent_kind: str,
):
    output_dir.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(losses)
    ax.set_xlabel("Iteration")
    ax.set_ylabel("-MLL loss")
    ax.set_title(f"Training loss ({model_name})")
    fig.tight_layout()
    fig.savefig(output_dir / f"loss_curve_{model_name}.png", dpi=150)
    plt.close(fig)

    # Scatter comparison plot works regardless of ambient dimensionality.
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.scatter(test_y_denorm.cpu().numpy(), pred_mean_denorm.cpu().numpy(), s=15, alpha=0.7)
    mn = min(test_y_denorm.min().item(), pred_mean_denorm.min().item())
    mx = max(test_y_denorm.max().item(), pred_mean_denorm.max().item())
    ax.plot([mn, mx], [mn, mx], "k--", linewidth=1)
    ax.set_xlabel("Ground truth")
    ax.set_ylabel("Predicted mean")
    ax.set_title(f"Prediction vs truth (original scale, {model_name})")
    fig.tight_layout()
    fig.savefig(output_dir / f"pred_vs_truth_{model_name}.png", dpi=150)
    plt.close(fig)

    if train_x.shape[1] == 2 and W_true.shape[1] >= 1:
        fig = plt.figure(figsize=(9, 7))
        ax = fig.add_subplot(111, projection="3d")

        all_x = torch.cat([train_x, test_x], dim=0)
        x_min, _ = all_x.min(dim=0)
        x_max, _ = all_x.max(dim=0)
        margin = 0.3
        x1 = torch.linspace(x_min[0] - margin, x_max[0] + margin, 60, device=all_x.device, dtype=all_x.dtype)
        x2 = torch.linspace(x_min[1] - margin, x_max[1] + margin, 60, device=all_x.device, dtype=all_x.dtype)
        xx, yy = torch.meshgrid(x1, x2, indexing="xy")
        grid_xy = torch.stack([xx.reshape(-1), yy.reshape(-1)], dim=-1)

        latent_fn = make_latent_function(latent_kind)
        zz_true = latent_fn(grid_xy @ W_true).reshape(xx.shape)

        ax.plot_surface(
            xx.detach().cpu().numpy(),
            yy.detach().cpu().numpy(),
            zz_true.detach().cpu().numpy(),
            cmap="viridis",
            alpha=0.55,
            linewidth=0.0,
            antialiased=True,
        )

        ax.scatter(
            train_x[:, 0].detach().cpu().numpy(),
            train_x[:, 1].detach().cpu().numpy(),
            train_y_denorm.detach().cpu().numpy(),
            c="tab:blue",
            s=20,
            alpha=0.9,
            label="Train data",
        )
        ax.scatter(
            test_x[:, 0].detach().cpu().numpy(),
            test_x[:, 1].detach().cpu().numpy(),
            test_y_denorm.detach().cpu().numpy(),
            c="tab:orange",
            s=20,
            alpha=0.9,
            label="Test data",
        )

        z_low = min(train_y_denorm.min().item(), test_y_denorm.min().item(), zz_true.min().item())
        z_high = max(train_y_denorm.max().item(), test_y_denorm.max().item(), zz_true.max().item())
        z_range = np.linspace(z_low, z_high, 40)
        xy_radius = float(torch.linalg.vector_norm(x_max - x_min).item())

        w_true_2d = W_true[:2, 0]
        if torch.linalg.vector_norm(w_true_2d) > 0:
            w_true_2d = w_true_2d / torch.linalg.vector_norm(w_true_2d)
            t_true = torch.linspace(-xy_radius, xy_radius, 2, device=w_true_2d.device, dtype=w_true_2d.dtype)
            s_true = torch.tensor(z_range, device=w_true_2d.device, dtype=w_true_2d.dtype)
            tt_true, ss_true = torch.meshgrid(t_true, s_true, indexing="xy")
            px_true = (tt_true * w_true_2d[0]).detach().cpu().numpy()
            py_true = (tt_true * w_true_2d[1]).detach().cpu().numpy()
            pz_true = ss_true.detach().cpu().numpy()
            ax.plot_surface(px_true, py_true, pz_true, color="tab:green", alpha=0.20, linewidth=0)

        if hasattr(model, "W") and model.W.shape[1] >= 1:
            w_est_2d = model.W.detach()[:2, 0]
            if torch.linalg.vector_norm(w_est_2d) > 0:
                w_est_2d = w_est_2d / torch.linalg.vector_norm(w_est_2d)
                t_est = torch.linspace(-xy_radius, xy_radius, 2, device=w_est_2d.device, dtype=w_est_2d.dtype)
                s_est = torch.tensor(z_range, device=w_est_2d.device, dtype=w_est_2d.dtype)
                tt_est, ss_est = torch.meshgrid(t_est, s_est, indexing="xy")
                px_est = (tt_est * w_est_2d[0]).detach().cpu().numpy()
                py_est = (tt_est * w_est_2d[1]).detach().cpu().numpy()
                pz_est = ss_est.detach().cpu().numpy()
                ax.plot_surface(px_est, py_est, pz_est, color="tab:red", alpha=0.20, linewidth=0)

        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_zlabel("output")
        ax.set_title("Ground-truth surface, data, and true/estimated 1D subspace planes")

        legend_handles = [
            plt.Line2D([0], [0], marker="o", color="w", markerfacecolor="tab:blue", markersize=8, label="Train data"),
            plt.Line2D([0], [0], marker="o", color="w", markerfacecolor="tab:orange", markersize=8, label="Test data"),
            plt.Line2D([0], [0], color="teal", linewidth=6, alpha=0.5, label="Ground-truth surface"),
            plt.Line2D([0], [0], color="tab:green", linewidth=6, alpha=0.4, label="True 1D subspace plane"),
        ]
        if hasattr(model, "W") and model.W.shape[1] >= 1:
            legend_handles.append(
                plt.Line2D([0], [0], color="tab:red", linewidth=6, alpha=0.4, label="Estimated 1D subspace plane")
            )
        ax.legend(handles=legend_handles, loc="upper left")

        fig.tight_layout()
        fig.savefig(output_dir / f"subspace_visualization_{model_name}.png", dpi=150)
        plt.close(fig)
    else:
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))

        def _to_2d_coords(arr: torch.Tensor) -> np.ndarray:
            cols = min(2, arr.shape[-1])
            out = torch.zeros(arr.shape[0], 2, device=arr.device, dtype=arr.dtype)
            out[:, :cols] = arr[:, :cols]
            return out.cpu().numpy()

        y_np = test_y_denorm.detach().cpu().numpy()
        z_true = test_x @ W_true
        pts_true = _to_2d_coords(z_true)
        ax_true = axes[0]
        sc_true = ax_true.scatter(pts_true[:, 0], pts_true[:, 1], c=y_np, s=14, cmap="viridis")
        ax_true.set_xlabel("true z1")
        ax_true.set_ylabel("true z2")
        ax_true.set_title("Ground-truth subspace (color = true y)")
        fig.colorbar(sc_true, ax=ax_true, shrink=0.7, pad=0.1)

        if hasattr(model, "W"):
            z_est = test_x @ model.W.detach()
        else:
            z_est = test_x[:, : W_true.shape[-1]]
        pts_est = _to_2d_coords(z_est)
        ax_est = axes[1]
        sc_est = ax_est.scatter(pts_est[:, 0], pts_est[:, 1], c=y_np, s=14, cmap="plasma")
        ax_est.set_xlabel("estimated z1")
        ax_est.set_ylabel("estimated z2")
        ax_est.set_title("Estimated linear subspace (color = true y)")
        fig.colorbar(sc_est, ax=ax_est, shrink=0.7, pad=0.1)

        fig.tight_layout()
        fig.savefig(output_dir / f"subspace_visualization_{model_name}.png", dpi=150)
        plt.close(fig)

    if hasattr(model, "W"):
        fig, ax = plt.subplots(figsize=(5, 4))
        est_basis, _ = torch.linalg.qr(model.W.detach())
        overlap = (W_true.transpose(-2, -1) @ est_basis).abs().cpu().numpy()
        im = ax.imshow(overlap, cmap="magma", vmin=0.0, vmax=1.0)
        ax.set_xlabel("Estimated basis index")
        ax.set_ylabel("True basis index")
        ax.set_title("Subspace overlap |W_true^T W_est|")
        fig.colorbar(im, ax=ax, label="Absolute overlap")
        fig.tight_layout()
        fig.savefig(output_dir / f"subspace_overlap_{model_name}.png", dpi=150)
        plt.close(fig)


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    device = torch.device("cuda" if args.device == "cuda" and torch.cuda.is_available() else "cpu")
    if args.device == "cuda" and device.type != "cuda":
        print("CUDA requested but not available; falling back to CPU.")

    train_x, train_y_raw, test_x, test_y_raw, W_true = make_linear_subspace_train_test_split(
        args.n_train,
        args.n_test,
        args.D,
        args.k_true,
        args.noise_std,
        latent_kind=args.latent,
        device=device,
    )
    train_y_mean = train_y_raw.mean()
    train_y_std = train_y_raw.std(unbiased=False).clamp_min(1e-6)
    train_y = (train_y_raw - train_y_mean) / train_y_std
    test_y = (test_y_raw - train_y_mean) / train_y_std

    likelihood = gpytorch.likelihoods.GaussianLikelihood().to(device)
    model_cls = MODEL_REGISTRY[args.model]
    model_kwargs = dict(
        train_x=train_x,
        train_y=train_y,
        likelihood=likelihood,
        input_dim=args.D,
        subspace_dim=args.k_model,
    )
    if args.model == "composite":
        model_kwargs["eps_alpha"] = args.eps_alpha

    model = model_cls(**model_kwargs).to(device)

    losses = train_model(model, likelihood, train_x, train_y, iters=args.iters, lr=args.lr)
    rmse, nll, pred_mean, _ = evaluate_model(model, likelihood, test_x, test_y)
    pred_mean_denorm = pred_mean * train_y_std + train_y_mean
    rmse_denorm = torch.sqrt(torch.mean((pred_mean_denorm - test_y_raw) ** 2)).item()

    print(f"Model: {args.model}")
    print(f"RMSE (normalized target): {rmse:.4f}")
    print(f"RMSE (original target scale): {rmse_denorm:.4f}")
    print(f"NLL / sample (normalized target): {nll:.4f}")

    if hasattr(model, "W"):
        est_W = model.W.detach()
        est_basis, _ = torch.linalg.qr(est_W)
        overlap = torch.linalg.matrix_norm(W_true.transpose(-2, -1) @ est_basis, ord=2).item()
        print(f"Subspace overlap spectral norm: {overlap:.4f}")
        if getattr(model, "uses_riemannian_projection", False):
            print(f"Orthogonality error (W): {orthogonality_error(est_W).item():.3e}")
    if hasattr(model, "eps"):
        print(f"Composite eps: {model.eps.item():.4f}")

    visualize_results(
        args.output_dir,
        args.model,
        model,
        train_x,
        train_y_raw,
        test_x,
        test_y_raw,
        pred_mean_denorm,
        losses,
        W_true,
        args.latent,
    )
    print(f"Saved artifacts to: {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
