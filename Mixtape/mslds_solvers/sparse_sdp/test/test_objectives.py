import sys
sys.path.append("..")
import numpy as np
from utils import numerical_derivative
from objectives import *
from constraints import *

def test_tr():
    dims = [1, 5, 10]
    N_rand = 10
    tol = 1e-3
    eps = 1e-4
    for dim in dims:
        for i in range(N_rand):
            X = np.random.rand(dim, dim)
            val = trace_obj(X)
            grad = grad_trace_obj(X)
            num_grad = numerical_derivative(trace_obj, X, eps)
            assert np.sum(np.abs(grad - num_grad)) < tol

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
