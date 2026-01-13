# Experiment 2: Noise comparison by batch size
# - Fixed learning rate for all methods: lr = 1e-4
# - Compare the first 50 parameter updates (zoomed-in view) to highlight noise
# - X-axis: Iterations (Updates), Y-axis: MSE Loss
#
# Expected qualitative behavior (noise level):
#   SGD (batch=1) > Mini-batch (batch=16) > GD (batch=N)

import os
import numpy as np

from src.data import load_fish_dataframe, make_train_matrix
from src.losses import mse_loss
from src.optimizers import step_gd, step_minibatch
from src.plotting import plot_loss_histories


def run_exp2(
    csv_path: str = "data/Fish.csv",
    lr: float = 1e-4,
    updates: int = 50,
    seed: int = 42,
    mb_batch_size: int = 16,
    save_path: str = "results/exp2_noise_vs_batch.png",
):
    """
    Runs Experiment 2 (noise vs batch size) with fixed learning rate.

    Parameters
    ----------
    csv_path : str
        Path to Fish.csv (downloaded from Kaggle; not included in the repo).
    lr : float
        Fixed learning rate applied to GD/SGD/Mini-batch (poster setting: 1e-4).
    updates : int
        Number of parameter updates to record (poster zoom: 0~50).
    seed : int
        Random seed for reproducibility (initialization + sampling).
    mb_batch_size : int
        Mini-batch size for Mini-batch SGD (poster setting: 16).
    save_path : str
        Where to save the plot image.
    """
    # Load and preprocess data (train split only is used for these experiments).
    df = load_fish_dataframe(csv_path, allow_dummy=False)
    X, y, _, _ = make_train_matrix(df)
    N, D = X.shape

    rng = np.random.RandomState(seed)

    # Use the same initialization for all methods for a fair comparison.
    theta0 = rng.randn(D, 1) * 0.01

    # -------------------------
    # GD: each "update" is one full-batch gradient step
    # -------------------------
    theta_gd = theta0.copy()
    loss_gd = []
    for _ in range(updates):
        theta_gd = step_gd(X, y, theta_gd, lr)
        loss_gd.append(mse_loss(X, y, theta_gd))

    # -------------------------
    # SGD (batch=1): each "update" samples one data point
    # -------------------------
    theta_sgd = theta0.copy()
    loss_sgd = []
    for _ in range(updates):
        idx = rng.randint(0, N)
        xi = X[idx : idx + 1]
        yi = y[idx : idx + 1]
        theta_sgd = step_minibatch(xi, yi, theta_sgd, lr)
        loss_sgd.append(mse_loss(X, y, theta_sgd))

    # -------------------------
    # Mini-batch SGD (batch=16): each "update" samples a random mini-batch
    # -------------------------
    theta_mb = theta0.copy()
    loss_mb = []
    for _ in range(updates):
        batch_idx = rng.choice(N, size=min(mb_batch_size, N), replace=False)
        xi = X[batch_idx]
        yi = y[batch_idx]
        theta_mb = step_minibatch(xi, yi, theta_mb, lr)
        loss_mb.append(mse_loss(X, y, theta_mb))

    # Ensure output directory exists
    out_dir = os.path.dirname(save_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    histories = {
        f"GD (lr={lr:.1g}, batch={N})": loss_gd,
        f"SGD (lr={lr:.1g}, batch=1)": loss_sgd,
        f"Mini-Batch (lr={lr:.1g}, batch={min(mb_batch_size, N)})": loss_mb,
    }

    plot_loss_histories(
        histories,
        title="Loss Stability: GD vs SGD vs Mini-Batch SGD",
        save_path=save_path,
    )


if __name__ == "__main__":
    run_exp2()
