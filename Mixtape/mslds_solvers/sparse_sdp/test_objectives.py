import numpy as np
from utils import numerical_derivative
from objectives import *

def test_sum_squares():
    dim = 1
    N_rand = 10
    tol = 1e-3
    eps = 1e-4
    for i in range(N_rand):
        X = np.random.rand(dim, dim)
        val = neg_sum_squares(X)
        grad = grad_neg_sum_squares(X)
        num_grad = numerical_derivative(neg_sum_sqares, X, eps)
        assert np.abs(grad - num_grad) < tol
