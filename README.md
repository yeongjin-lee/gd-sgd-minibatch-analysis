## Abstract

This repository presents an empirical comparison of three gradient-based
optimization methods — Gradient Descent (GD), Stochastic Gradient Descent (SGD),
and Mini-batch SGD — on a multivariate linear regression task using a real-world dataset.

All optimizers were implemented from scratch in Python. We evaluate their
convergence behavior in terms of loss reduction speed and optimization noise
under different learning rates and batch sizes. The experiments demonstrate
the trade-off between stability and efficiency across optimization methods,
highlighting why mini-batch SGD is widely used in modern machine learning.
