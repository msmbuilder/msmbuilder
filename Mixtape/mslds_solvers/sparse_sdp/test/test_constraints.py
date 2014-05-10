import sys
sys.path.append("..")
from constraints import *
from utils import numerical_derivative
import numpy as np

def test_quadratic_inequality():
    """
    Test quadratic inequality specification.
    """
    dim, As, bs, Cs, ds, Fs, gradFs, Gs, gradGs = \
            quadratic_inequality()
    tol = 1e-3
    eps = 1e-4
    N_rand = 10
    for (f, gradf) in zip(Fs, gradFs):
        for i in range(N_rand):
            X = np.random.rand(dim, dim)
            val = f(X)
            grad = gradf(X)
            print "grad:\n", grad
            num_grad = numerical_derivative(f, X, eps)
            print "num_grad:\n", num_grad
            assert np.sum(np.abs(grad - num_grad)) < tol

def test_quadratic_equality():
    """
    Test quadratic equality specification.
    """
    dim, As, bs, Cs, ds, Fs, gradFs, Gs, gradGs = \
            quadratic_equality()
    tol = 1e-3
    eps = 1e-4
    N_rand = 10
    for (g, gradg) in zip(Gs, gradGs):
        for i in range(N_rand):
            X = np.random.rand(dim, dim)
            val = g(X)
            grad = gradg(X)
            print "grad:\n", grad
            num_grad = numerical_derivative(g, X, eps)
            print "num_grad:\n", num_grad
            assert np.sum(np.abs(grad - num_grad)) < tol
