import sys
sys.path.append("..")
import numpy as np
from utils import numerical_derivative
from objectives import *

def test_sum_squares():
    dims = [1, 5, 10]
    N_rand = 10
    tol = 1e-3
    eps = 1e-4
    for dim in dims:
        for i in range(N_rand):
            X = np.random.rand(dim, dim)
            val = neg_sum_squares(X)
            grad = grad_neg_sum_squares(X)
            num_grad = numerical_derivative(neg_sum_squares, X, eps)
            assert np.sum(np.abs(grad - num_grad)) < tol
