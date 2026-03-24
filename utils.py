"""Utilities for optimization on the Stiefel manifold."""

from __future__ import annotations

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
