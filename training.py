"""Training helper mixing Euclidean Adam with Stiefel manifold Adam."""

from __future__ import annotations

from typing import Iterable, Optional

import torch

from manifold_optim import RiemannianAdam
from utils import orthogonality_error


def train_epoch_with_stiefel(
    model: torch.nn.Module,
    data_loader: Iterable,
    loss_fn,
    W: torch.nn.Parameter,
    *,
    manifold_lr: float = 1e-3,
    euclidean_lr: float = 1e-3,
    betas=(0.9, 0.999),
    eps: float = 1e-8,
    weight_decay: float = 0.0,
    device: Optional[torch.device] = None,
    verbose: bool = False,
) -> float:
    """Train one epoch with separate optimizers for manifold and non-manifold params.

    Non-manifold (e.g., GP hyperparameters) are updated using standard Adam,
    while ``W`` is updated with ``RiemannianAdam`` on Stiefel.
    """
    euclidean_params = [p for p in model.parameters() if p is not W and p.requires_grad]
    euclidean_optim = torch.optim.Adam(
        euclidean_params,
        lr=euclidean_lr,
        betas=betas,
        eps=eps,
        weight_decay=weight_decay,
    )
    manifold_optim = RiemannianAdam(
        [W],
        lr=manifold_lr,
        betas=betas,
        eps=eps,
        weight_decay=weight_decay,
    )

    model.train()
    running_loss = 0.0
    n_batches = 0

    for batch in data_loader:
        if device is not None:
            batch = tuple(x.to(device) for x in batch)

        euclidean_optim.zero_grad(set_to_none=True)
        manifold_optim.zero_grad(set_to_none=True)

        loss = loss_fn(model, batch)
        loss.backward()

        euclidean_optim.step()
        manifold_optim.step()

        running_loss += float(loss.detach())
        n_batches += 1

        if verbose:
            ortho = orthogonality_error(W).item()
            print(f"loss={float(loss.detach()):.6f} | orthogonality_error={ortho:.3e}")

    return running_loss / max(n_batches, 1)
