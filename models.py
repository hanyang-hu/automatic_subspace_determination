"""GP model definitions for full-space, projected, and composite kernels."""

from __future__ import annotations

import math
from typing import Optional

import gpytorch
import torch
from botorch.models.gpytorch import GPyTorchModel
from gpytorch.constraints import Positive
from gpytorch.priors import HalfCauchyPrior


DEFAULT_EPS_ALPHA = 0.01


def _initialize_kernel_lengthscale(kernel: gpytorch.kernels.Kernel, dim: int) -> None:
    """Initialize RBF lengthscales to sqrt(dim), matching report instructions."""
    if hasattr(kernel, "lengthscale"):
        kernel.lengthscale = math.sqrt(dim)


class ProjectedKernel(gpytorch.kernels.Kernel):
    """Kernel wrapper that projects inputs before applying a base kernel."""

    has_lengthscale = False

    def __init__(
        self,
        input_dim: int,
        subspace_dim: int,
        base_kernel: Optional[gpytorch.kernels.Kernel] = None,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        if subspace_dim > input_dim:
            raise ValueError("subspace_dim must be <= input_dim")

        self.input_dim = input_dim
        self.subspace_dim = subspace_dim
        self.base_kernel = base_kernel or gpytorch.kernels.RBFKernel(
            ard_num_dims=subspace_dim,
            lengthscale_constraint=Positive(),
        )
        _initialize_kernel_lengthscale(self.base_kernel, subspace_dim)

        self.register_parameter(
            name="W",
            parameter=torch.nn.Parameter(self._stiefel_init(input_dim, subspace_dim)),
        )

    @staticmethod
    def _stiefel_init(input_dim: int, subspace_dim: int) -> torch.Tensor:
        """Initialize W on the Stiefel manifold via QR decomposition."""
        q, _ = torch.linalg.qr(torch.randn(input_dim, subspace_dim))
        return q[:, :subspace_dim]

    def project(self, x: torch.Tensor) -> torch.Tensor:
        if x.size(-1) != self.input_dim:
            raise ValueError(
                f"Expected input dimension {self.input_dim}, got {x.size(-1)}"
            )
        
        # x = x.to(self.W.dtype)  # Ensure input is same dtype as W for matmul

        return x @ self.W

    def forward(
        self,
        x1: torch.Tensor,
        x2: torch.Tensor,
        diag: bool = False,
        **params,
    ) -> torch.Tensor:
        x1_proj = self.project(x1)
        x2_proj = self.project(x2)
        return self.base_kernel(x1_proj, x2_proj, diag=diag, **params)


class LinearEmbeddingKernel(ProjectedKernel):
    """Projected kernel with an unconstrained linear embedding matrix."""

    @staticmethod
    def _stiefel_init(input_dim: int, subspace_dim: int) -> torch.Tensor:
        # Reuse parent construction path but initialize unconstrained weights.
        scale = 1.0 / math.sqrt(max(input_dim, 1))
        return torch.randn(input_dim, subspace_dim) * scale


class CompositeKernel(gpytorch.kernels.Kernel):
    """Composite kernel with projected and residual full-space terms."""

    has_lengthscale = False

    def __init__(
        self,
        input_dim: int,
        subspace_dim: int,
        projected_base_kernel: Optional[gpytorch.kernels.Kernel] = None,
        residual_kernel: Optional[gpytorch.kernels.Kernel] = None,
        eps_alpha: float = DEFAULT_EPS_ALPHA,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.projected_kernel = ProjectedKernel(
            input_dim=input_dim,
            subspace_dim=subspace_dim,
            base_kernel=projected_base_kernel,
        )
        self.residual_kernel = residual_kernel or gpytorch.kernels.RBFKernel(
            ard_num_dims=input_dim,
            lengthscale_constraint=Positive(),
        )
        _initialize_kernel_lengthscale(self.residual_kernel, input_dim)

        self._ensure_raw_eps()
        # print("Initialized CompositeKernel with eps_alpha =", eps_alpha)
        # Initialize eps with the alpha value to encourage starting with a meaningful contribution from the residual kernel.
        self._set_eps(eps_alpha)
        # print("Initial eps value set to:", self.eps.item())
        self.register_prior(
            "eps_prior",
            HalfCauchyPrior(scale=eps_alpha),
            lambda m: m.eps,
            lambda m, v: m._set_eps(v),
        )

    def _ensure_raw_eps(self) -> None:
        """Ensure ``raw_eps`` exists for compatibility with older objects/state."""
        if "raw_eps" not in self._parameters:
            self.register_parameter(
                name="raw_eps", parameter=torch.nn.Parameter(torch.tensor(0.0))
            )
        if "raw_eps_constraint" not in self._constraints:
            self.register_constraint("raw_eps", Positive())

    @property
    def eps(self) -> torch.Tensor:
        self._ensure_raw_eps()
        return self.raw_eps_constraint.transform(self.raw_eps)

    def _set_eps(self, value: torch.Tensor | float) -> None:
        self._ensure_raw_eps()
        if not torch.is_tensor(value):
            value = torch.as_tensor(
                value, dtype=self.raw_eps.dtype, device=self.raw_eps.device
            )
        self.initialize(raw_eps=self.raw_eps_constraint.inverse_transform(value))

    @eps.setter
    def eps(self, value: torch.Tensor | float) -> None:
        self._set_eps(value)

    @property
    def W(self) -> torch.nn.Parameter:
        return self.projected_kernel.W

    def forward(
        self,
        x1: torch.Tensor,
        x2: torch.Tensor,
        diag: bool = False,
        **params,
    ) -> torch.Tensor:
        projected_term = self.projected_kernel(x1, x2, diag=diag, **params)
        residual_term = self.residual_kernel(x1, x2, diag=diag, **params)
        return projected_term + self.eps * residual_term


class _BaseExactGP(gpytorch.models.ExactGP, GPyTorchModel):
    """Shared setup for exact GP models in this repository."""
    _num_outputs = 1

    def __init__(
        self,
        train_x: torch.Tensor,
        train_y: torch.Tensor,
        likelihood: gpytorch.likelihoods.GaussianLikelihood,
    ) -> None:
        super().__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ConstantMean()

        # Keep positivity on observation noise.
        self.likelihood.noise_covar.register_constraint("raw_noise", Positive())
        self.likelihood.noise_covar.noise = 1e-4  # Initialize to small noise for stability.

    def forward(self, x: torch.Tensor) -> gpytorch.distributions.MultivariateNormal:
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)


class StandardGPModel(_BaseExactGP):
    """Exact GP with a standard full-space kernel."""

    def __init__(
        self,
        train_x: torch.Tensor,
        train_y: torch.Tensor,
        likelihood: gpytorch.likelihoods.GaussianLikelihood,
        input_dim: int,
        subspace_dim: int,
    ) -> None:
        super().__init__(train_x, train_y, likelihood)
        del subspace_dim  # Kept for constructor consistency across model choices.

        base_kernel = gpytorch.kernels.RBFKernel(
            ard_num_dims=input_dim,
            lengthscale_constraint=Positive(),
        )
        _initialize_kernel_lengthscale(base_kernel, input_dim)
        self.covar_module = gpytorch.kernels.ScaleKernel(
            base_kernel,
            outputscale_constraint=Positive(),
        )
        self.to(dtype=train_x.dtype, device=train_x.device)


class ProjectedGPModel(_BaseExactGP):
    """Exact GP operating purely in a learned projected subspace."""

    def __init__(
        self,
        train_x: torch.Tensor,
        train_y: torch.Tensor,
        likelihood: gpytorch.likelihoods.GaussianLikelihood,
        input_dim: int,
        subspace_dim: int,
    ) -> None:
        super().__init__(train_x, train_y, likelihood)
        self.uses_riemannian_projection = True

        self.covar_module = gpytorch.kernels.ScaleKernel(
            ProjectedKernel(input_dim=input_dim, subspace_dim=subspace_dim),
            outputscale_constraint=Positive(),
        )
        self.to(dtype=train_x.dtype, device=train_x.device)

    @property
    def W(self) -> torch.nn.Parameter:
        return self.covar_module.base_kernel.W


class CompositeGPModel(_BaseExactGP):
    """Exact GP with projected + epsilon-weighted full-space residual kernel."""

    def __init__(
        self,
        train_x: torch.Tensor,
        train_y: torch.Tensor,
        likelihood: gpytorch.likelihoods.GaussianLikelihood,
        input_dim: int,
        subspace_dim: int,
        eps_alpha: float = DEFAULT_EPS_ALPHA,
    ) -> None:
        super().__init__(train_x, train_y, likelihood)
        self.uses_riemannian_projection = True

        self.covar_module = gpytorch.kernels.ScaleKernel(
            CompositeKernel(
                input_dim=input_dim,
                subspace_dim=subspace_dim,
                eps_alpha=eps_alpha,
            ),
            outputscale_constraint=Positive(),
        )
        self.to(dtype=train_x.dtype, device=train_x.device)

    @property
    def W(self) -> torch.nn.Parameter:
        return self.covar_module.base_kernel.W

    def _ensure_raw_eps(self) -> None:
        """Ensure ``raw_eps`` exists for compatibility with older objects/state."""
        if "raw_eps" not in self._parameters:
            self.register_parameter(
                name="raw_eps", parameter=torch.nn.Parameter(torch.tensor(0.0))
            )
        if "raw_eps_constraint" not in self._constraints:
            self.register_constraint("raw_eps", Positive())

    @property
    def eps(self) -> torch.Tensor:
        return self.covar_module.base_kernel.eps


class LinearEmbeddingGPModel(_BaseExactGP):
    """Exact GP using an unconstrained learned linear embedding ``xW``."""

    def __init__(
        self,
        train_x: torch.Tensor,
        train_y: torch.Tensor,
        likelihood: gpytorch.likelihoods.GaussianLikelihood,
        input_dim: int,
        subspace_dim: int,
    ) -> None:
        super().__init__(train_x, train_y, likelihood)
        self.uses_riemannian_projection = False

        self.covar_module = gpytorch.kernels.ScaleKernel(
            LinearEmbeddingKernel(input_dim=input_dim, subspace_dim=subspace_dim),
            outputscale_constraint=Positive(),
        )
        self.to(dtype=train_x.dtype, device=train_x.device)

    @property
    def W(self) -> torch.nn.Parameter:
        return self.covar_module.base_kernel.W
