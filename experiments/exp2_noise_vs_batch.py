# Experiment 2: Loss stability (noise) comparison
# - Fixed learning rate for all methods: lr = 1e-4
# - Train size fixed to 126 (to match the original experiment)
# - Use numeric features only (drop Species) to match the original code path
# - Record loss after each parameter update (SGD/MB have many updates),
#   and plot only the first 50 updates (zoom-in view)

import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from src.plotting import plot_loss_histories


def _load_numeric_only(csv_path: str):
    """
    Loads Fish.csv and returns:
      X: numeric features [Length1, Length2, Length3, Height, Width]
      y: target Weight
    This matches the original experiment that did not use 'Species'.
    """
    df = pd.read_csv(csv_path)
    df = df[df["Weight"] > 0]

    y = df["Weight"].to_numpy().reshape(-1, 1)
    X = df[["Length1", "Length2", "Length3", "Height", "Width"]].to_numpy()
    return X, y


def _standardize_train_only(X_train, X_test):
    """Manual standardization to match the original implementation."""
    mean = X_train.mean(axis=0)
    std = X_train.std(axis=0)
    std[std == 0] = 1.0
    return (X_train - mean) / std, (X_test - mean) / std


def _mse_half(y_pred, y_true):
    """0.5 * mean squared error (matches the original)."""
    diff = y_pred - y_true
    return float(0.5 * np.mean(diff ** 2))


def run_exp2(
    csv_path: str = "data/Fish.csv",
    lr: float = 1e-4,
    epochs: int = 50,
    train_size: int = 126,
    mb_batch_size: int = 16,
    seed: int = 42,
    zoom_updates: int = 50,
    save_path: str = "results/exp2_noise_vs_batch.png",
):
    # Load dataset (numeric-only)
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"'{csv_path}' not found. See data/README.md for download instructions.")

    X, y = _load_numeric_only(csv_path)
    y = y.reshape(-1)  # original code uses 1D y

    # Train/test split with fixed train size (matches original)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, train_size=train_size, random_state=seed
    )

    # Standardize using training stats only
    X_train, X_test = _standardize_train_only(X_train, X_test)

    N, D = X_train.shape
    rng = np.random.RandomState(seed)

    # Initialize weights as zeros (matches original)
    w0 = np.zeros(D, dtype=float)
    b0 = 0.0

    # ---------- GD (full batch): 1 update per epoch ----------
    w_gd = w0.copy()
    b_gd = float(b0)
    loss_gd = []
    for _ in range(epochs):
        y_pred = X_train @ w_gd + b_gd
        diff = y_pred - y_train
        gw = (X_train.T @ diff) / N
        gb = float(np.mean(diff))
        w_gd -= lr * gw
        b_gd -= lr * gb
        loss_gd.append(_mse_half(X_train @ w_gd + b_gd, y_train))

    # ---------- SGD (batch=1): many updates ----------
    w_sgd = w0.copy()
    b_sgd = float(b0)
    loss_sgd = []
    for _ in range(epochs):
        idx = rng.permutation(N)
        for i in range(0, N, 1):
            bi = idx[i:i+1]
            Xb = X_train[bi]
            yb = y_train[bi]
            y_pred = Xb @ w_sgd + b_sgd
            diff = y_pred - yb
            gw = (Xb.T @ diff) / len(bi)
            gb = float(np.mean(diff))
            w_sgd -= lr * gw
            b_sgd -= lr * gb
            loss_sgd.append(_mse_half(X_train @ w_sgd + b_sgd, y_train))

    # ---------- Mini-batch (batch=16): many updates ----------
    w_mb = w0.copy()
    b_mb = float(b0)
    loss_mb = []
    for _ in range(epochs):
        idx = rng.permutation(N)
        for i in range(0, N, mb_batch_size):
            bi = idx[i:i+mb_batch_size]
            Xb = X_train[bi]
            yb = y_train[bi]
            y_pred = Xb @ w_mb + b_mb
            diff = y_pred - yb
            gw = (Xb.T @ diff) / len(bi)
            gb = float(np.mean(diff))
            w_mb -= lr * gw
            b_mb -= lr * gb
            loss_mb.append(_mse_half(X_train @ w_mb + b_mb, y_train))

    # Zoom-in: plot only the first 50 updates (matches xlim(-1,49) intent)
    histories = {
        f"GD (lr={lr:.1g}, batch={N})": loss_gd[:zoom_updates],
        f"SGD (lr={lr:.1g}, batch=1)": loss_sgd[:zoom_updates],
        f"Mini-batch (lr={lr:.1g}, batch={mb_batch_size})": loss_mb[:zoom_updates],
    }

    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)

    plot_loss_histories(
        histories,
        title="Loss Stability: GD vs SGD vs Mini-Batch SGD",
        save_path=save_path,
    )


if __name__ == "__main__":
    run_exp2()
