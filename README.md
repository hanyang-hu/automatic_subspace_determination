# Automatic Subspace Determination with GPyTorch

This repository provides Gaussian Process (GP) models for subspace-aware regression using **GPyTorch**, including:

- **Standard GP** (`StandardGPModel`): full-space RBF kernel.
- **Projected GP** (`ProjectedGPModel`): kernel on learned projected coordinates `xW`.
- **Linear Embedding GP** (`LinearEmbeddingGPModel`): kernel on learned `xW` with unconstrained Euclidean optimization of `W`.
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
python test_model.py --model composite --D 2 --k_true 1 --k_model 1 --n_train 200 --n_test 200 --iters 300 --noise_std 0.1
```

Switch models:

```bash
python test_model.py --model standard
python test_model.py --model projected
python test_model.py --model linear_embedding
python test_model.py --model composite
```

Optional flags:

- `--latent {linear,smooth,nonlinear}`: latent response function in the true subspace.
- `--device {cpu,cuda}`: run on GPU when available.
- `--output_dir artifacts`: folder for output plots.

## Data generation in `test_model.py`

- Inputs are generated once with Sobol quasi-random samples on `[-2.5, 2.5]^2`, then randomly split into train/test.
- Train and test therefore come from the same distribution (not disjoint regions).
- Targets are generated from a **1D true subspace** (`k_true=1`) in the 2D input and normalized with train-set mean/std before GP fitting.
- Metrics report both normalized-scale RMSE and original-scale RMSE.

## Expected outputs

The script saves plots under `artifacts/`:

- `loss_curve_<model>.png`
- `pred_vs_truth_<model>.png`
- `subspace_visualization_<model>.png`
- `subspace_overlap_<model>.png` (for models that learn `W`)

and prints summary metrics:

- RMSE
- NLL per sample
- orthogonality error for learned projection matrix `W` (if present)
- overlap diagnostic with true subspace (if present)

## Notes

- `W` is initialized on the Stiefel manifold using QR for `projected/composite`, and with unconstrained Gaussian weights for `linear_embedding`.
- `RiemannianAdam` projects Adam directions onto the tangent space and retracts with QR.
- A Half-Cauchy prior is attached only to the composite gating parameter `eps` (default scale `alpha=0.01`).


## Synthetic high-dimensional Bayesian optimization benchmark suite

Use `run_synthetic_bo_benchmark.py` to compare the four GP models (`standard`, `projected`, `linear_embedding`, `composite`) on:

- 3 synthetic BoTorch functions with non-axis-aligned low-dimensional embeddings (`Ackley`, `Rastrigin`, `Levy`)

The script runs repeated BO with seeds `41-45` by default, saves per-seed traces to CSV (including raw observed values and incumbent values), prints progress every 10 iterations, and plots mean best-observed performance over iterations.

```bash
python run_synthetic_bo_benchmark.py
```

Useful options:

- `--output_dir outputs`
- `--synthetic_ambient_dim 50`
- `--n_init 50`
- `--n_iter 100`
- `--train_steps 50`
- `--lr 0.05`
- `--stagnation_patience 20`
- `--print_every 10`
- `--seeds 41 42 43 44 45`
- `--model_subspace_dim 20`: learned subspace dimension for `projected`, `linear_embedding`, and `composite` models.

### Compare model mean traces for one synthetic benchmark

After running BO, you can compare only the **mean** trajectories across models (no std shading):

```bash
python visualize_synthetic_bo_means.py --benchmark_output_dir outputs/synthetic_levy
```

This writes:

- `outputs/synthetic_levy/bo_model_mean_comparison.png`

### Subspace alignment vs dataset size (Ackley 6D in 50D)

Use `run_synthetic_subspace_alignment.py` to study how many Sobol-sampled data points are needed to recover the true subspace, with warm-starting as dataset size grows.

```bash
python run_synthetic_subspace_alignment.py \
  --output_dir outputs/subspace_alignment_ackley \
  --models projected linear_embedding composite \
  --n_start 50 --n_step 25 --n_steps 8 \
  --num_repeats 5 --seeds 41 42 43 44 45 \
  --subspace_dim 6 --latent_dim 6
```

Outputs:

- `alignment_all_runs.csv`
- `alignment_aggregated.csv`
- `alignment_vs_dataset_size.png`

Outputs (for each benchmark/model pair):

- `outputs/<benchmark>/<model>/bo_run_seed_<seed>.csv`
- `outputs/<benchmark>/<model>/bo_results_all_runs.csv`
- `outputs/<benchmark>/<model>/bo_results_aggregated.csv`
- `outputs/<benchmark>/<model>/bo_results_plot.png`
