# Empirical Analysis of Gradient-Based Optimization Methods

## Abstract

This repository presents an empirical comparison of three gradient-based
optimization algorithms — **Gradient Descent (GD)**, **Stochastic Gradient Descent (SGD)**,
and **Mini-batch SGD** — on a multivariate linear regression task using a real-world dataset
(Fish Market, Kaggle).

All optimizers are implemented **from scratch in NumPy**, while `scikit-learn` is used only
for data preprocessing (standardization and one-hot encoding). We analyze the optimization
behavior from two complementary perspectives:

1. **Convergence speed** — each method is run with its own automatically tuned learning rate
   (selected by minimizing the average training loss), and the full loss reduction trajectories
   are compared.
2. **Optimization noise** — the learning rate is fixed across methods, and loss fluctuations
   are examined over the first 50 parameter updates to highlight the effect of batch size on
   stability.

Our experiments empirically demonstrate the classic trade-off between **efficiency and stability**:
SGD exhibits the highest noise due to single-sample updates, Mini-batch SGD reduces this variance
through partial averaging, and GD produces the smoothest but most computationally expensive
optimization trajectory.

Rather than focusing on exact numerical replication of any single figure, this repository
aims to provide a **clean, reproducible, and interpretable experimental framework**
for understanding gradient-based optimization dynamics.

## Project Structure
- `src/optimizers.py`: Core NumPy implementations of GD, SGD, and Mini-batch SGD update rules.
- `src/lr_search.py`: Learning-rate search utility (grid search over log-spaced candidates).
- `src/data.py`: Dataset loading and preprocessing (sklearn used only for splitting/encoding/scaling).
- `src/losses.py`: MSE loss implementation.
- `src/plotting.py`: Plotting utilities for loss curves.

- `experiments/exp1_loss_decrease.py`: **Experiment 1** — Convergence trajectories with automatically tuned learning rates.
- `experiments/exp2_noise_vs_batch.py`: **Experiment 2** — Loss stability (noise) comparison with fixed learning rate over the first 50 updates.
- `main.py`: Entry point to run experiments end-to-end.

