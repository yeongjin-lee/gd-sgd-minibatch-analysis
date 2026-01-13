import numpy as np

def grad_full_batch(X, y, theta):
    m = len(y)
    return (1/m) * X.T.dot(X.dot(theta) - y)

def step_gd(X, y, theta, lr):
    theta = theta - lr * grad_full_batch(X, y, theta)
    return theta

def step_minibatch(xi, yi, theta, lr):
    m_batch = len(yi)
    grad = (1/m_batch) * xi.T.dot(xi.dot(theta) - yi)
    return theta - lr * grad

def shuffled_batches(X, y, batch_size, rng=None):
    if rng is None:
        rng = np.random
    N = X.shape[0]
    idx = rng.permutation(N)
    Xs = X[idx]
    ys = y[idx]
    for i in range(0, N, batch_size):
        j = min(i + batch_size, N)
        yield Xs[i:j], ys[i:j]
