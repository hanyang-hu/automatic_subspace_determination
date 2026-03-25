# Automatic Subspace Determination with GPyTorch

This repository provides Gaussian Process (GP) models for subspace-aware regression using **GPyTorch**, including:

- **Standard GP** (`StandardGPModel`): full-space RBF kernel.
- **Projected GP** (`ProjectedGPModel`): kernel on learned projected coordinates `xW`.
- **Composite GP** (`CompositeGPModel`): projected kernel plus an `eps`-weighted full-space residual kernel.

It also includes a **Riemannian Adam** optimizer for Stiefel-constrained projection matrices using tangent projection + QR retraction.

## Files

- `models.py`: GPyTorch model and kernel implementations with positivity constraints and Half-Cauchy priors.
- `utils.py`: Stiefel geometry helpers + synthetic data generation utilities.
- `manifold_optim.py`: `RiemannianAdam` optimizer implementation.
- `test_model.py`: CLI script for synthetic experiments, metrics, target normalization, and visualization.

## Installation

```bash
conda create -n asd-gp python=3.11 -y
conda activate asd-gp
pip install -r requirements.txt
```

## Run synthetic modeling test

Example with the composite model:

```bash
python test_model.py --model composite --D 5 --k_true 2 --k_model 2 --n_train 200 --n_test 200 --iters 300 --noise_std 0.1
```

Switch models:

```bash
python test_model.py --model standard
python test_model.py --model projected
python test_model.py --model composite
```

Optional flags:

- `--latent {linear,smooth,nonlinear}`: latent response function in the true subspace.
- `--device {cpu,cuda}`: run on GPU when available.
- `--output_dir artifacts`: folder for output plots.

## Data generation in `test_model.py`

- Inputs are generated once with Sobol quasi-random samples on `[-2.5, 2.5]^D`, then randomly split into train/test.
- Train and test therefore come from the same distribution (not disjoint regions).
- Targets are normalized with train-set mean/std before GP fitting.
- Metrics report both normalized-scale RMSE and original-scale RMSE.

## Expected outputs

The script saves plots under `artifacts/`:

- `loss_curve_<model>.png`
- `pred_vs_truth_<model>.png`
- `objective_and_subspace_response_<model>.png`
- `subspace_overlap_<model>.png` (for models that learn `W`)

and prints summary metrics:

- RMSE
- NLL per sample
- orthogonality error for learned projection matrix `W` (if present)
- overlap diagnostic with true subspace (if present)

## Notes

- `W` is initialized on the Stiefel manifold using QR.
- `RiemannianAdam` projects Adam directions onto the tangent space and retracts with QR.
- A Half-Cauchy prior is attached only to the composite gating parameter `eps` (default scale `alpha=0.1`).
