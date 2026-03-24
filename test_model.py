"""CLI script to train and visualize GP subspace models on synthetic data."""

from __future__ import annotations

import argparse
from pathlib import Path

import gpytorch
import matplotlib.pyplot as plt
import torch

from manifold_optim import RiemannianAdam
from models import CompositeGPModel, ProjectedGPModel, StandardGPModel
from utils import make_linear_subspace_data, orthogonality_error, set_seed


MODEL_REGISTRY = {
    "standard": StandardGPModel,
    "projected": ProjectedGPModel,
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
    parser.add_argument("--eps_alpha", type=float, default=0.1, help="Half-Cauchy prior scale for composite eps")
    return parser.parse_args()


def split_parameter_groups(model: torch.nn.Module) -> tuple[list[torch.nn.Parameter], list[torch.nn.Parameter]]:
    manifold_params: list[torch.nn.Parameter] = []
    if hasattr(model, "W"):
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
def visualize_results(output_dir: Path, model_name: str, model, train_x, train_y, test_x, test_y, pred_mean, losses):
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

    # For D>=2, visualize a 2D response slice with remaining dimensions fixed to 0.
    if test_x.shape[-1] >= 2:
        grid_n = 60
        x1 = torch.linspace(-2.5, 2.5, grid_n, device=test_x.device)
        x2 = torch.linspace(-2.5, 2.5, grid_n, device=test_x.device)
        xx1, xx2 = torch.meshgrid(x1, x2, indexing="ij")

        x_grid = torch.zeros(grid_n * grid_n, test_x.shape[-1], device=test_x.device)
        x_grid[:, 0] = xx1.reshape(-1)
        x_grid[:, 1] = xx2.reshape(-1)

        model.eval()
        pred_grid = model(x_grid).mean.reshape(grid_n, grid_n).cpu().numpy()

        fig, ax = plt.subplots(figsize=(6, 5))
        pcm = ax.pcolormesh(xx1.cpu().numpy(), xx2.cpu().numpy(), pred_grid, shading="auto")
        fig.colorbar(pcm, ax=ax, label="Predicted response")
        ax.set_xlabel("x[0]")
        ax.set_ylabel("x[1]")
        ax.set_title(f"Predicted response surface slice ({model_name})")
        fig.tight_layout()
        fig.savefig(output_dir / f"surface_slice_{model_name}.png", dpi=150)
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

    if hasattr(model, "W"):
        est_W = model.W.detach()
        overlap = torch.linalg.matrix_norm(W_true.transpose(-2, -1) @ est_W, ord=2).item()
        print(f"Subspace overlap spectral norm: {overlap:.4f}")
        print(f"Orthogonality error (W): {orthogonality_error(est_W).item():.3e}")
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
    )
    print(f"Saved artifacts to: {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
