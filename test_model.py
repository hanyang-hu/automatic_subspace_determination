"""CLI script to train and visualize GP subspace models on synthetic data."""

from __future__ import annotations

import argparse
from pathlib import Path

import gpytorch
import matplotlib.pyplot as plt
import numpy as np
import torch

from manifold_optim import RiemannianAdam
from models import CompositeGPModel, ProjectedGPModel, StandardGPModel
from utils import make_linear_subspace_train_test_split, orthogonality_error, set_seed


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
def visualize_results(
    output_dir: Path,
    model_name: str,
    model,
    test_x,
    test_y_denorm,
    pred_mean_denorm,
    losses,
    W_true,
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

    # Visualize objective value + response in true/estimated subspaces.
    fig = plt.figure(figsize=(12, 8))
    gs = fig.add_gridspec(2, 2)
    ax_loss = fig.add_subplot(gs[0, :])
    ax_loss.plot(losses)
    ax_loss.set_xlabel("Iteration")
    ax_loss.set_ylabel("-MLL loss")
    ax_loss.set_title(f"Objective function trajectory ({model_name})")

    def _to_3d_coords(arr: torch.Tensor) -> np.ndarray:
        cols = min(2, arr.shape[-1])
        out = torch.zeros(arr.shape[0], 3, device=arr.device, dtype=arr.dtype)
        out[:, :cols] = arr[:, :cols]
        return out.cpu().numpy()

    z_true = test_x @ W_true
    pts_true = _to_3d_coords(z_true)
    pts_true[:, 2] = test_y_denorm.detach().cpu().numpy()
    ax_true = fig.add_subplot(gs[1, 0], projection="3d")
    sc_true = ax_true.scatter(pts_true[:, 0], pts_true[:, 1], pts_true[:, 2], c=pts_true[:, 2], s=12, cmap="viridis")
    ax_true.set_xlabel("true z1")
    ax_true.set_ylabel("true z2")
    ax_true.set_zlabel("y")
    ax_true.set_title("Ground-truth response on true subspace")
    fig.colorbar(sc_true, ax=ax_true, shrink=0.7, pad=0.1)

    if hasattr(model, "W"):
        z_est = test_x @ model.W.detach()
    else:
        z_est = test_x[:, : W_true.shape[-1]]
    pts_est = _to_3d_coords(z_est)
    pts_est[:, 2] = pred_mean_denorm.detach().cpu().numpy()
    ax_est = fig.add_subplot(gs[1, 1], projection="3d")
    sc_est = ax_est.scatter(pts_est[:, 0], pts_est[:, 1], pts_est[:, 2], c=pts_est[:, 2], s=12, cmap="plasma")
    ax_est.set_xlabel("estimated z1")
    ax_est.set_ylabel("estimated z2")
    ax_est.set_zlabel("pred y")
    ax_est.set_title("Predicted response on estimated subspace")
    fig.colorbar(sc_est, ax=ax_est, shrink=0.7, pad=0.1)

    fig.tight_layout()
    fig.savefig(output_dir / f"objective_and_subspace_response_{model_name}.png", dpi=150)
    plt.close(fig)

    if hasattr(model, "W"):
        fig, ax = plt.subplots(figsize=(5, 4))
        overlap = (W_true.transpose(-2, -1) @ model.W.detach()).abs().cpu().numpy()
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
        overlap = torch.linalg.matrix_norm(W_true.transpose(-2, -1) @ est_W, ord=2).item()
        print(f"Subspace overlap spectral norm: {overlap:.4f}")
        print(f"Orthogonality error (W): {orthogonality_error(est_W).item():.3e}")
    if hasattr(model, "eps"):
        print(f"Composite eps: {model.eps.item():.4f}")

    visualize_results(
        args.output_dir,
        args.model,
        model,
        test_x,
        test_y_raw,
        pred_mean_denorm,
        losses,
        W_true,
    )
    print(f"Saved artifacts to: {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
