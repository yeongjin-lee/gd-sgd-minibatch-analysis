# Understanding Gradient Noise: An Empirical Analysis of GD, SGD, and Mini-batch SGD

## 1. Motivation
While gradient-based optimization is fundamental to machine learning, practical behaviors such as training instability
and gradient noise are often discussed only qualitatively.

This project empirically investigates how batch size directly influences the stability and efficiency of optimization
in a controlled setting. We compare three standard methods:

- Gradient Descent (GD)
- Stochastic Gradient Descent (SGD)
- Mini-batch SGD

## 2. Experimental Setup
We fit a multivariate Linear Regression model ($\hat{y} = Xw + b$) on the Fish Market dataset (Kaggle)
to predict fish weight from physical measurements.

- Objective: Minimize Mean Squared Error (MSE)

$$\mathcal{L}(w,b) = \frac{1}{N}\sum_{i=1}^{N} (\hat{y}_i - y_i)^2$$

- Implementation: All optimizers are implemented from scratch using NumPy to explicitly control the update dynamics and ensure full transparency of the optimization process.

## 3. Experiment 1 — Convergence Speed
- Goal: Compare pure optimization efficiency.
- Method: Each optimizer is run with its own automatically tuned learning rate, selected via grid search to minimize the average training loss.
- Observation: This experiment isolates how quickly each method can reduce the objective when its hyperparameters are idealized.

## 4. Experiment 2 — Noise & Stability
- Goal: Visualize the stochastic noise induced by different batch sizes.
- Method: The learning rate is fixed across all methods, while only the batch size is varied:
  - GD: $N=126$ (Full batch)
  - Mini-batch SGD: $B=16$
  - SGD: B = 1
- Observation: We analyze the loss fluctuations over the first 50 parameter updates.
  - SGD: High-variance, noisy updates.
  - Mini-batch: Partial averaging reduces noise.
  - GD: Smooth, deterministic trajectory.
 
## 5. Key Takeaway
These experiments reveal the fundamental trade-off between exploration and stability:
| Method | Convergence / Exploration | Stability (Noise) |
| :--- | :--- | :--- |
| **SGD** | Aggressive (High variance) | Low |
| **Mini-batch SGD** | **Balanced** | **Medium** |
| **GD** | Conservative | High |
