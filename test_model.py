"""CLI script to train and visualize GP subspace models on synthetic data."""

from __future__ import annotations

import argparse
from pathlib import Path

import gpytorch
import matplotlib.pyplot as plt
import torch

from manifold_optim import RiemannianAdam
from models import (
    DEFAULT_EPS_ALPHA,
    CompositeGPModel,
    LinearEmbeddingGPModel,
    ProjectedGPModel,
    StandardGPModel,
)
from utils import make_linear_subspace_data, orthogonality_error, set_seed


MODEL_REGISTRY = {
    "standard": StandardGPModel,
    "projected": ProjectedGPModel,
    "linear_embed": LinearEmbeddingGPModel,
    "composite": CompositeGPModel,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=MODEL_REGISTRY, default="composite")
    parser.add_argument("--D", type=int, default=5, help="Ambient input dimension")
    parser.add_argument("--k_true", type=int, default=2, help="True latent subspace dim")
    parser.add_argument("--k_model", type=int, default=2, help="Model latent subspace dim")
    parser.add_argument("--n_train", type=int, default=200)
    parser.add_argument("--n_test", type=int, default=200)
    parser.add_argument("--noise_std", type=float, default=0.1)
    parser.add_argument("--iters", type=int, default=300)
    parser.add_argument("--lr", type=float, default=5e-2)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    parser.add_argument("--latent", default="smooth", choices=["linear", "smooth", "nonlinear"])
    parser.add_argument("--output_dir", type=Path, default=Path("artifacts"))
    parser.add_argument(
        "--eps_alpha",
        type=float,
        default=DEFAULT_EPS_ALPHA,
        help="Half-Cauchy prior scale for composite eps",
    )
    return parser.parse_args()


def split_parameter_groups(model: torch.nn.Module) -> tuple[list[torch.nn.Parameter], list[torch.nn.Parameter]]:
    if hasattr(model, "manifold_parameters"):
        manifold_params = list(model.manifold_parameters())
    elif hasattr(model, "W") and getattr(model, "is_stiefel", False):
        manifold_params = [model.W]
    else:
        manifold_params = []

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
    train_y,
    test_x,
    test_y,
    pred_mean,
    losses,
    W_true: torch.Tensor,
    est_subspace: torch.Tensor | None,
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
    ax.scatter(test_y.cpu().numpy(), pred_mean.cpu().numpy(), s=15, alpha=0.7)
    mn = min(test_y.min().item(), pred_mean.min().item())
    mx = max(test_y.max().item(), pred_mean.max().item())
    ax.plot([mn, mx], [mn, mx], "k--", linewidth=1)
    ax.set_xlabel("Ground truth")
    ax.set_ylabel("Predicted mean")
    ax.set_title(f"Prediction vs truth ({model_name})")
    fig.tight_layout()
    fig.savefig(output_dir / f"pred_vs_truth_{model_name}.png", dpi=150)
    plt.close(fig)

    # Visualize true vs estimated subspace bases.
    if est_subspace is not None:
        fig, axes = plt.subplots(1, 2, figsize=(9, 4), sharey=True)
        im0 = axes[0].imshow(W_true.cpu().numpy(), aspect="auto", cmap="coolwarm")
        axes[0].set_title("Ground-truth subspace basis")
        axes[0].set_xlabel("Basis index")
        axes[0].set_ylabel("Ambient dimension")
        fig.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)

        im1 = axes[1].imshow(est_subspace.cpu().numpy(), aspect="auto", cmap="coolwarm")
        axes[1].set_title("Estimated subspace basis")
        axes[1].set_xlabel("Basis index")
        fig.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)

        fig.tight_layout()
        fig.savefig(output_dir / f"subspace_bases_{model_name}.png", dpi=150)
        plt.close(fig)

    # Visualize model response on true/estimated 2D subspace coordinates.
    if W_true.shape[-1] >= 2 and est_subspace is not None and est_subspace.shape[-1] >= 2:
        grid_n = 60
        z1 = torch.linspace(-2.5, 2.5, grid_n, device=test_x.device)
        z2 = torch.linspace(-2.5, 2.5, grid_n, device=test_x.device)
        zz1, zz2 = torch.meshgrid(z1, z2, indexing="ij")
        z_grid = torch.stack([zz1.reshape(-1), zz2.reshape(-1)], dim=-1)

        x_true_grid = z_grid @ W_true[:, :2].transpose(-2, -1)
        x_est_grid = z_grid @ est_subspace[:, :2].transpose(-2, -1)

        model.eval()
        pred_true_grid = model(x_true_grid).mean.reshape(grid_n, grid_n).cpu().numpy()
        pred_est_grid = model(x_est_grid).mean.reshape(grid_n, grid_n).cpu().numpy()

        fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), sharex=True, sharey=True)
        pcm0 = axes[0].pcolormesh(zz1.cpu().numpy(), zz2.cpu().numpy(), pred_true_grid, shading="auto")
        fig.colorbar(pcm0, ax=axes[0], label="Predicted response")
        axes[0].set_xlabel("z[0]")
        axes[0].set_ylabel("z[1]")
        axes[0].set_title("Response on ground-truth subspace")

        pcm1 = axes[1].pcolormesh(zz1.cpu().numpy(), zz2.cpu().numpy(), pred_est_grid, shading="auto")
        fig.colorbar(pcm1, ax=axes[1], label="Predicted response")
        axes[1].set_xlabel("z[0]")
        axes[1].set_ylabel("z[1]")
        axes[1].set_title("Response on estimated subspace")
        fig.tight_layout()
        fig.savefig(output_dir / f"surface_true_vs_est_{model_name}.png", dpi=150)
        plt.close(fig)


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    device = torch.device("cuda" if args.device == "cuda" and torch.cuda.is_available() else "cpu")
    if args.device == "cuda" and device.type != "cuda":
        print("CUDA requested but not available; falling back to CPU.")

    train_x, train_y, W_true = make_linear_subspace_data(
        args.n_train,
        args.D,
        args.k_true,
        args.noise_std,
        latent_kind=args.latent,
        device=device,
    )
    test_x, test_y, _ = make_linear_subspace_data(
        args.n_test,
        args.D,
        args.k_true,
        args.noise_std,
        latent_kind=args.latent,
        device=device,
    )

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

    print(f"Model: {args.model}")
    print(f"RMSE: {rmse:.4f}")
    print(f"NLL / sample: {nll:.4f}")

    est_subspace = None
    if hasattr(model, "W"):
        est_W = model.W.detach()
        if est_W.shape[-1] > 0:
            est_subspace, _ = torch.linalg.qr(est_W, mode="reduced")
            est_subspace = est_subspace[:, : est_W.shape[-1]]
        overlap = torch.linalg.matrix_norm(W_true.transpose(-2, -1) @ est_subspace, ord=2).item()
        print(f"Subspace overlap spectral norm: {overlap:.4f}")
        if getattr(model, "is_stiefel", False):
            print(f"Orthogonality error (W): {orthogonality_error(est_W).item():.3e}")
        else:
            print("Using unconstrained linear embedding (non-Stiefel optimization).")
    if hasattr(model, "eps"):
        print(f"Composite eps: {model.eps.item():.4f}")

    visualize_results(
        args.output_dir,
        args.model,
        model,
        train_x,
        train_y,
        test_x,
        test_y,
        pred_mean,
        losses,
        W_true,
        est_subspace,
    )
    print(f"Saved artifacts to: {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
