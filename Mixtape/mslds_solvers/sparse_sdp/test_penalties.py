import numpy as np
from constraints import *
from penalties import *
from utils import numerical_derivative

def test1a():
    """
    Check gradients of log-sum-exp on simple equality constraint
    problem.
    """
    dim, As, bs, Cs, ds, Fs, gradFs, Gs, gradGs = \
           simple_equality_constraint()
    tol = 1e-3
    M = compute_scale(len(As), len(Cs), len(Fs), len(Gs), tol)
    def f(X):
        return log_sum_exp_penalty(X, M, As, bs, Cs, ds, Fs, Gs)
    def gradf(X):
        return log_sum_exp_grad_penalty(X, M, As,
                    bs, Cs, ds, Fs, gradFs, Gs, gradGs)
    eps = 1e-4
    N_rand = 10
    for i in range(N_rand):
        X = np.random.rand(dim, dim)
        val = f(X)
        grad = gradf(X)
        print "grad:\n", grad
        num_grad = numerical_derivative(f, X, eps)
        print "num_grad:\n", num_grad
        assert np.sum(np.abs(grad - num_grad)) < tol

def test2():
    """
    Check log-sum-exp gradients on simple equality and inequaliy
    constrained problem.
    """
    dim, As, bs, Cs, ds, Fs, gradFs, Gs, gradGs = \
           simple_equality_and_inequality_constraint()
    tol = 1e-3
    M = compute_scale(len(As), len(Cs), len(Fs), len(Gs), tol)
    def f(X):
        return log_sum_exp_penalty(X, M, As, bs, Cs, ds, Fs, Gs)
    def gradf(X):
        return log_sum_exp_grad_penalty(X, M, As,
                    bs, Cs, ds, Fs, gradFs, Gs, gradGs)
    eps = 1e-4
    N_rand = 10
    for i in range(N_rand):
        X = np.random.rand(dim, dim)
        val = f(X)
        grad = gradf(X)
        print "grad:\n", grad
        num_grad = numerical_derivative(f, X, eps)
        print "num_grad:\n", num_grad
        assert np.sum(np.abs(grad - num_grad)) < tol

def test1b():
    """
    Check gradients of neg_max on simple equality constraint problem.
    TODO: Think about how to test correctness of the drawn subgradient.
    """
    dim, As, bs, Cs, ds, Fs, gradFs, Gs, gradGs = \
           simple_equality_constraint()
    tol = 1e-3
    M = compute_scale(len(As), len(Cs), len(Fs), len(Gs), tol)
    def f(X):
        return neg_max_penalty(X, M, As, bs, Cs, ds, Fs, Gs)
    def gradf(X):
        return neg_max_grad_penalty(X, M, As,
                    bs, Cs, ds, Fs, gradFs, Gs, gradGs)
    assert True == False

