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

This repository provides **fully reproducible scripts** for both experiments and is intended
as a minimal, research-oriented implementation for understanding gradient-based optimization
dynamics.
