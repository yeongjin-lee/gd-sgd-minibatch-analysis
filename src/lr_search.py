import numpy as np
from .losses import mse_loss
from .optimizers import grad_full_batch, step_minibatch, shuffled_batches

def find_best_lr(method_name, lr_candidates, X, y, epochs=50, batch_size=1, seed=42):
    best_lr = None
    best_loss = float('inf')

    N, D = X.shape
    initial_theta = np.random.randn(D, 1) * 0.01

    print(f"[{method_name}] searching lr...", end=" ")

    for lr in lr_candidates:
        theta = initial_theta.copy()
        np.random.seed(seed)

        lr_losses = []

        for _ in range(epochs):
            if method_name == "gd":
                gradient = (1/N) * X.T.dot(X.dot(theta) - y)
                theta -= lr * gradient
            else:
                bs = 1 if method_name == "sgd" else batch_size
                for xi, yi in shuffled_batches(X, y, bs):
                    theta = step_minibatch(xi, yi, theta, lr)

            lr_losses.append(mse_loss(X, y, theta))

        avg_loss = float('inf') if np.isnan(lr_losses).any() else float(np.mean(lr_losses))

        if avg_loss < best_loss:
            best_loss = avg_loss
            best_lr = lr

    print(f"-> best={best_lr:.5g} (avg loss={best_loss:.4g})")
    return best_lr
