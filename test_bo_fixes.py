from __future__ import annotations

import gpytorch
import torch

from manifold_optim import RiemannianAdam
from models import CompositeGPModel, ProjectedKernel
from run_synthetic_bo_benchmark import build_warm_start, fit_surrogate, pick_next_point


def _toy_data(n: int = 24, dim: int = 8) -> tuple[torch.Tensor, torch.Tensor]:
    torch.manual_seed(7)
    x = torch.rand(n, dim, dtype=torch.double)
    y = torch.sin(x[:, :1] * 3.0).sum(dim=-1)
    return x, y


def test_composite_warm_start_preserves_kernel_and_eps_and_noise() -> None:
    train_x, train_y = _toy_data()
    model, likelihood = fit_surrogate(
        model_name="composite",
        train_x=train_x,
        train_y=train_y,
        subspace_dim=3,
        train_steps=8,
        lr=0.03,
        manifold_lr_mult=1.0,
        warm_start=None,
        warm_start_w_only=False,
    )
    warm = build_warm_start(model=model, likelihood=likelihood, include_non_w=True)

    model2, likelihood2 = fit_surrogate(
        model_name="composite",
        train_x=train_x,
        train_y=train_y,
        subspace_dim=3,
        train_steps=0,
        lr=0.03,
        manifold_lr_mult=1.0,
        warm_start=warm,
        warm_start_w_only=False,
    )
    assert torch.allclose(model.W, model2.W, atol=1e-12)
    assert torch.allclose(model.covar_module.outputscale, model2.covar_module.outputscale, atol=1e-12)
    assert torch.allclose(model.covar_module.base_kernel.eps, model2.covar_module.base_kernel.eps, atol=1e-12)
    assert torch.allclose(
        model.covar_module.base_kernel.residual_kernel.lengthscale,
        model2.covar_module.base_kernel.residual_kernel.lengthscale,
        atol=1e-12,
    )
    assert torch.allclose(likelihood.noise, likelihood2.noise, atol=1e-12)


def test_riemannian_step_keeps_w_on_stiefel() -> None:
    train_x, train_y = _toy_data()
    likelihood = gpytorch.likelihoods.GaussianLikelihood()
    model = CompositeGPModel(
        train_x=train_x,
        train_y=train_y,
        likelihood=likelihood,
        input_dim=train_x.shape[-1],
        subspace_dim=3,
    )
    w0 = model.W.detach().clone()
    model.W.grad = torch.randn_like(model.W)
    opt = RiemannianAdam([model.W], lr=0.05)
    opt.step()
    gram = model.W.transpose(-2, -1) @ model.W
    ident = torch.eye(gram.shape[-1], dtype=gram.dtype, device=gram.device)
    assert torch.linalg.matrix_norm(gram - ident) < 1e-6
    assert not torch.allclose(w0, model.W)


def test_projected_kernel_handles_dtype_device_cast() -> None:
    ker = ProjectedKernel(input_dim=5, subspace_dim=2).to(dtype=torch.double)
    x = torch.randn(6, 5, dtype=torch.float32)
    z = ker.project(x)
    assert z.dtype == ker.W.dtype
    assert z.device == ker.W.device


def test_latent_acquisition_mode_returns_valid_candidate() -> None:
    train_x, train_y = _toy_data()
    model, _ = fit_surrogate(
        model_name="composite",
        train_x=train_x,
        train_y=train_y,
        subspace_dim=3,
        train_steps=6,
        lr=0.03,
        manifold_lr_mult=1.0,
        warm_start=None,
        warm_start_w_only=False,
    )
    x_next = pick_next_point(
        model=model,
        dim=train_x.shape[-1],
        best_f=float(train_y.max().item()),
        num_restarts=4,
        raw_samples=32,
        optimize_in_latent=True,
    )
    assert x_next.shape == (1, train_x.shape[-1])
    assert torch.all(x_next >= 0.0) and torch.all(x_next <= 1.0)
