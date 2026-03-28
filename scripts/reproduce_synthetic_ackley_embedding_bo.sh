#!/usr/bin/env bash
set -euo pipefail

# Reproduce projected/composite BO behavior on synthetic_ackley with latent-space
# acquisition enabled and full hyperparameter warm start.
python run_synthetic_bo_benchmark.py \
  --models standard projected composite linear_embedding \
  --synthetic_ambient_dim 50 \
  --n_init 30 \
  --n_iter 80 \
  --train_steps 40 \
  --lr 0.04 \
  --manifold_lr_mult 1.0 \
  --warm_start_mode all \
  --acq_opt_space auto \
  --seeds 41 42 \
  --output_dir outputs/repro_ackley
