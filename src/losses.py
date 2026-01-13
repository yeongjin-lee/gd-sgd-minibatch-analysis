import numpy as np

def mse_loss(X, y, theta):
    """
    Computes the Mean Squared Error (MSE) loss for linear regression.

    This implementation follows the standard definition:
        L(θ) = (1 / (2m)) * Σ (Xθ − y)^2

    where:
        X     : input matrix (with bias term)
        y     : target vector
        theta : parameter vector
        m     : number of samples
    """
    m = len(y)
    if m == 0:
        return 0.0

    predictions = X.dot(theta)
    return float((1 / (2 * m)) * np.sum((predictions - y) ** 2))
