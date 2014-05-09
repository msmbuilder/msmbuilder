"""
Implementation of various useful utility functions.
"""
import numpy as np

# Convenience matrix access functions
def set_entries(X, coords, Z):
    x_low, x_hi, y_low, y_hi = coords
    X[x_low:x_hi, y_low:y_hi] = Z

def get_entries(X, coords):
    x_low, x_hi, y_low, y_hi = coords
    return X[x_low:x_hi, y_low:y_hi]

def numerical_derivative(f, X, eps):
    """
    Numerical gradient of a matrix valued function that accepts
    dim by dim real matrices as arguments. Uses formula

    grad f[i,j] \approx (f(X + eps e_ij) - f(X - eps e_ij))/ (2 eps)
    """
    (dim, _) = np.shape(X)
    grad = np.zeros((dim, dim))
    for i in range(dim):
        for j in range(dim):
            # Calculate upper
            X[i,j] += eps
            f_ij_plus = f(X)
            X[i,j] -= eps

            # Calculate lower
            X[i,j] -= eps
            f_ij_minus = f(X)
            X[i,j] += eps
            grad[i,j] = (f_ij_plus - f_ij_minus)/(2*eps)
    return grad
