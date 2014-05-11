import sys
sys.path.append("..")
import numpy as np
from constraints import *
from penalties import *
from utils import numerical_derivative

def test1():
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

def test3():
    """
    Check log-sum-exp gradient on quadratic inequality problem.
    """
    dim, As, bs, Cs, ds, Fs, gradFs, Gs, gradGs = \
           quadratic_inequality()
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

def test4():
    """
    Check log-sum-exp gradient on quadratic equality problem.
    """
    dim, As, bs, Cs, ds, Fs, gradFs, Gs, gradGs = \
           quadratic_equality()
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

def test5():
    """
    Check log-sum-exp gradient on many linear inequalities.
    """
    tol = 1e-3
    eps = 1e-4
    N_rand = 10
    dims = [4, 8]
    for dim in dims:
        As, bs, Cs, ds, Fs, gradFs, Gs, gradGs = \
                stress_inequalities(dim)
        M = compute_scale(len(As), len(Cs), len(Fs), len(Gs), tol)
        def f(X):
            return log_sum_exp_penalty(X, M, As, bs, Cs, ds, Fs, Gs)
        def gradf(X):
            return log_sum_exp_grad_penalty(X, M, As,
                        bs, Cs, ds, Fs, gradFs, Gs, gradGs)
        for i in range(N_rand):
            X = np.random.rand(dim, dim)
            val = f(X)
            grad = gradf(X)
            print "grad:\n", grad
            num_grad = numerical_derivative(f, X, eps)
            print "num_grad:\n", num_grad
            assert np.sum(np.abs(grad - num_grad)) < tol

def test6():
    """
    Check log-sum-exp gradient on many linear equalities.
    """
    tol = 1e-3
    eps = 1e-4
    N_rand = 10
    dims = [4, 16]
    for dim in dims:
        As, bs, Cs, ds, Fs, gradFs, Gs, gradGs = \
                stress_equalities(dim)
        M = compute_scale(len(As), len(Cs), len(Fs), len(Gs), tol)
        def f(X):
            return log_sum_exp_penalty(X, M, As, bs, Cs, ds, Fs, Gs)
        def gradf(X):
            return log_sum_exp_grad_penalty(X, M, As,
                        bs, Cs, ds, Fs, gradFs, Gs, gradGs)
        for i in range(N_rand):
            X = np.random.rand(dim, dim)
            val = f(X)
            grad = gradf(X)
            print "grad:\n", grad
            num_grad = numerical_derivative(f, X, eps)
            print "num_grad:\n", num_grad
            assert np.sum(np.abs(grad - num_grad)) < tol

def test7():
    """
    BROKEN: Check log-sum-exp gradient on many linear and nonlinear
    equalities.
    """
    tol = 1e-3
    eps = 1e-4
    N_rand = 10
    dims = [16]
    for dim in dims:
        As, bs, Cs, ds, Fs, gradFs, Gs, gradGs = \
                stress_inequalities_and_equalities(dim)
        M = compute_scale(len(As), len(Cs), len(Fs), len(Gs), tol)
        def f(X):
            return log_sum_exp_penalty(X, M, As, bs, Cs, ds, Fs, Gs)
        def gradf(X):
            return log_sum_exp_grad_penalty(X, M, As,
                        bs, Cs, ds, Fs, gradFs, Gs, gradGs)
        for i in range(N_rand):
            X = np.random.rand(dim, dim)
            val = f(X)
            grad = gradf(X)
            print "grad:\n", grad
            num_grad = numerical_derivative(f, X, eps)
            print "num_grad:\n", num_grad
            diff = np.sum(np.abs(grad - num_grad))
            print "diff: ", diff
            assert diff < tol

def test8():
    """
    Check log-sum-exp gradient on basic batch equalities
    """
    tol = 1e-3
    eps = 1e-5
    N_rand = 10
    dims = [16]
    for dim in dims:
        block_dim = int(dim/2)
        # Generate random configurations
        A = np.random.rand(block_dim, block_dim)
        B = np.random.rand(block_dim, block_dim)
        B = np.dot(B.T, B)
        D = np.random.rand(block_dim, block_dim)
        D = np.dot(D.T, D)
        tr_B_D = np.trace(B) + np.trace(D)
        B = B / tr_B_D
        D = D / tr_B_D
        As, bs, Cs, ds, Fs, gradFs, Gs, gradGs = \
                basic_batch_equality(dim, A, B, D)
        M = compute_scale(len(As), len(Cs), len(Fs), len(Gs), tol)
        def f(X):
            return log_sum_exp_penalty(X, M, As, bs, Cs, ds, Fs, Gs)
        def gradf(X):
            return log_sum_exp_grad_penalty(X, M, As,
                        bs, Cs, ds, Fs, gradFs, Gs, gradGs)
        for i in range(N_rand):
            X = np.random.rand(dim, dim)
            val = f(X)
            grad = gradf(X)
            print "grad:\n", grad
            num_grad = numerical_derivative(f, X, eps)
            print "num_grad:\n", num_grad
            diff = np.sum(np.abs(grad - num_grad))
            print "diff: ", diff
            assert diff < tol

#def test1b():
#    """
#    Check gradients of neg_max on simple equality constraint problem.
#    TODO: Think about how to test correctness of the drawn subgradient.
#    """
#    dim, As, bs, Cs, ds, Fs, gradFs, Gs, gradGs = \
#           simple_equality_constraint()
#    tol = 1e-3
#    M = compute_scale(len(As), len(Cs), len(Fs), len(Gs), tol)
#    def f(X):
#        return neg_max_penalty(X, M, As, bs, Cs, ds, Fs, Gs)
#    def gradf(X):
#        return neg_max_grad_penalty(X, M, As,
#                    bs, Cs, ds, Fs, gradFs, Gs, gradGs)
#    assert True == False
#
