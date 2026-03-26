"""Utilities for optimization and synthetic-data generation."""

from __future__ import annotations

from typing import Callable

import torch


def stiefel_tangent_projection(W: torch.Tensor, A: torch.Tensor) -> torch.Tensor:
    """Project ambient matrix ``A`` onto the tangent space at ``W`` on Stiefel.

    Uses the symmetric equivalent form:
        A - 0.5 * W (W^T A) - 0.5 * W (A^T W)
    """
    wt_a = W.transpose(-2, -1) @ A
    at_w = A.transpose(-2, -1) @ W
    return A - 0.5 * W @ wt_a - 0.5 * W @ at_w


def qr_retraction(W: torch.Tensor, Z: torch.Tensor) -> torch.Tensor:
    """Retract tangent perturbation ``Z`` back to Stiefel using QR.

    Returns the Q factor of ``W + Z`` with sign correction chosen so that
    the diagonal of ``R`` is non-negative.
    """
    Y = W + Z
    Q, R = torch.linalg.qr(Y, mode="reduced")

    diag = torch.diagonal(R, dim1=-2, dim2=-1)
    signs = torch.sign(diag)
    signs = torch.where(signs == 0, torch.ones_like(signs), signs)

    # Flip Q columns such that R has positive diagonal.
    Q = Q * signs.unsqueeze(-2)
    return Q


def orthogonality_error(W: torch.Tensor) -> torch.Tensor:
    """Return ||W^T W - I||_F as a diagnostic tensor."""
    n = W.shape[-1]
    ident = torch.eye(n, dtype=W.dtype, device=W.device)
    gram = W.transpose(-2, -1) @ W
    return torch.linalg.matrix_norm(gram - ident, ord="fro")


def set_seed(seed: int) -> None:
    """Set random seed for reproducibility."""
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def make_latent_function(kind: str = "smooth") -> Callable[[torch.Tensor], torch.Tensor]:
    """Return a latent response function defined on the projected coordinates."""
    if kind == "linear":
        return lambda z: z[..., 0]
    if kind == "smooth":
        return lambda z: torch.sin(2.0 * z[..., 0]) + 0.5 * torch.cos(3.0 * z[..., 0])
    if kind == "nonlinear":
        return lambda z: torch.sin(2.0 * z[..., 0]) + 0.3 * z[..., 0].square()
    raise ValueError(f"Unsupported latent function kind: {kind}")


def make_linear_subspace_data(
    n: int,
    input_dim: int,
    subspace_dim: int,
    noise_std: float,
    *,
    latent_kind: str = "smooth",
    device: torch.device,
    dtype: torch.dtype = torch.float32,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Sample x in R^D and y = g(W_true^T x) + eps with Stiefel-orthonormal W_true."""
    x = torch.randn(n, input_dim, device=device, dtype=dtype)
    q, _ = torch.linalg.qr(
        torch.randn(input_dim, subspace_dim, device=device, dtype=dtype),
        mode="reduced",
    )
    W_true = q[:, :subspace_dim]
    z = x @ W_true
    g = make_latent_function(latent_kind)
    y_clean = g(z)
    y = y_clean + noise_std * torch.randn_like(y_clean)
    return x, y, W_true


def make_linear_subspace_train_test_split(
    n_train: int,
    n_test: int,
    input_dim: int,
    subspace_dim: int,
    noise_std: float,
    *,
    latent_kind: str = "smooth",
    device: torch.device,
    dtype: torch.dtype = torch.float32,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Generate quasi-random data once, then split into train and test sets.

    Inputs are sampled with a Sobol engine and scaled to ``[-2.5, 2.5]^D``.
    Targets are generated from one shared ``W_true`` and latent function.
    """
    n_total = n_train + n_test
    sobol = torch.quasirandom.SobolEngine(dimension=input_dim, scramble=True)
    x_unit = sobol.draw(n_total).to(device=device, dtype=dtype)
    x = 5.0 * (x_unit - 0.5)

    q, _ = torch.linalg.qr(
        torch.randn(input_dim, subspace_dim, device=device, dtype=dtype),
        mode="reduced",
    )
    W_true = q[:, :subspace_dim]

    # print(latent_kind)
    if latent_kind != "nonlowdim":
        z = x @ W_true
        g = make_latent_function(latent_kind)
        y_clean = g(z)
    else:
        # For the nonlowdim case, we want to generate targets that depend on all input dimensions,
        y_clean = torch.sin(x).sum(dim=-1)

    y = y_clean + noise_std * torch.randn_like(y_clean)

    perm = torch.randperm(n_total, device=device)
    train_idx = perm[:n_train]
    test_idx = perm[n_train:]
    return x[train_idx], y[train_idx], x[test_idx], y[test_idx], W_true
