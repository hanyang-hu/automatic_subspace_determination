"""Riemannian optimization primitives for Stiefel-constrained parameters."""

from __future__ import annotations

from typing import Iterable, Tuple

import torch
from torch.optim import Optimizer

from utils import qr_retraction, stiefel_tangent_projection


class RiemannianAdam(Optimizer):
    """Adam on Stiefel manifold via tangent projection + QR retraction.

    First and second moments are tracked for Euclidean gradients, then the
    Adam direction is projected to the tangent space before retraction.
    """

    def __init__(
        self,
        params: Iterable[torch.nn.Parameter],
        lr: float = 1e-3,
        betas: Tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0.0,
    ) -> None:
        if lr <= 0.0:
            raise ValueError(f"Invalid lr: {lr}")
        if eps <= 0.0:
            raise ValueError(f"Invalid eps: {eps}")
        beta1, beta2 = betas
        if not (0.0 <= beta1 < 1.0 and 0.0 <= beta2 < 1.0):
            raise ValueError(f"Invalid betas: {betas}")
        if weight_decay < 0.0:
            raise ValueError(f"Invalid weight_decay: {weight_decay}")

        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = group["lr"]
            beta1, beta2 = group["betas"]
            eps = group["eps"]
            weight_decay = group["weight_decay"]

            for p in group["params"]:
                if p.grad is None:
                    continue
                if p.grad.is_sparse:
                    raise RuntimeError("RiemannianAdam does not support sparse gradients")

                grad = p.grad
                if weight_decay != 0:
                    grad = grad.add(p, alpha=weight_decay)

                state = self.state[p]
                if len(state) == 0:
                    state["step"] = 0
                    state["exp_avg"] = torch.zeros_like(p)
                    state["exp_avg_sq"] = torch.zeros_like(p)

                exp_avg = state["exp_avg"]
                exp_avg_sq = state["exp_avg_sq"]
                state["step"] += 1
                step_t = state["step"]

                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)

                bias_correction1 = 1 - beta1**step_t
                bias_correction2 = 1 - beta2**step_t

                denom = exp_avg_sq.sqrt() / (bias_correction2**0.5)
                denom.add_(eps)
                euclidean_direction = (exp_avg / bias_correction1) / denom

                tangent_direction = stiefel_tangent_projection(p, euclidean_direction)
                new_p = qr_retraction(p, -lr * tangent_direction)
                p.copy_(new_p)

        return loss


# Optional alias requested in task description.
StiefelAdam = RiemannianAdam
