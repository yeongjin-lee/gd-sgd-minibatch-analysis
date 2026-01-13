import numpy as np

from src.data import load_fish_dataframe, make_train_matrix
from src.lr_search import find_best_lr
from src.losses import mse_loss
from src.optimizers import step_gd, step_minibatch, shuffled_batches
from src.plotting import plot_loss_histories

def run_exp1(csv_path="data/Fish.csv", epochs=50, seed=42, save_path="results/exp1_loss.png"):
    df = load_fish_dataframe(csv_path, allow_dummy=False)
    X, y, _, _ = make_train_matrix(df)

    N, D = X.shape
    y_flat = y.flatten()

    batch_size_mb = 16

    lr_candidates_gd = np.logspace(-4, 0, 20)
    lr_candidates_mb = np.logspace(-4, 0, 20)
    lr_candidates_sgd = np.logspace(-5, -1, 20)

    print("\n--- LR search (avg loss) ---")
    best_lr_gd  = find_best_lr("gd",  lr_candidates_gd,  X, y, epochs=epochs, batch_size=N, seed=seed)
    best_lr_mb  = find_best_lr("mb",  lr_candidates_mb,  X, y, epochs=epochs, batch_size=batch_size_mb, seed=seed)
    best_lr_sgd = find_best_lr("sgd", lr_candidates_sgd, X, y, epochs=epochs, batch_size=1, seed=seed)

    # init
    np.random.seed(seed)
    theta0 = np.random.randn(D, 1) * 0.01

    # SGD
    theta = theta0.copy()
    loss_sgd = []
    np.random.seed(seed)
    for _ in range(epochs):
        for xi, yi in shuffled_batches(X, y_flat.reshape(-1,1), batch_size=1):
            theta = step_minibatch(xi, yi, theta, best_lr_sgd)
            loss_sgd.append(mse_loss(X, y, theta))

    # Mini-batch
    theta = theta0.copy()
    loss_mb = []
    np.random.seed(seed)
    for _ in range(epochs):
        for xi, yi in shuffled_batches(X, y_flat.reshape(-1,1), batch_size=batch_size_mb):
            theta = step_minibatch(xi, yi, theta, best_lr_mb)
            loss_mb.append(mse_loss(X, y, theta))

    # GD
    theta = theta0.copy()
    loss_gd = []
    np.random.seed(seed)
    for _ in range(epochs):
        theta = step_gd(X, y, theta, best_lr_gd)
        loss_gd.append(mse_loss(X, y, theta))

    histories = {
        f"GD (lr={best_lr_gd:.3g}, batch={N})": loss_gd,
        f"Mini-batch (lr={best_lr_mb:.3g}, batch={batch_size_mb})": loss_mb,
        f"SGD (lr={best_lr_sgd:.3g}, batch=1)": loss_sgd,
    }

    plot_loss_histories(histories, title="Loss Reduction: Auto-tuned by Average Loss", save_path=save_path)

if __name__ == "__main__":
    run_exp1()
