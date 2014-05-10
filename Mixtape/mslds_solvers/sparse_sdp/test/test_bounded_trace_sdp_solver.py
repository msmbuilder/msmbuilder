import sys
sys.path.append("..")
from bounded_trace_sdp_solver import BoundedTraceSolver
from objectives import neg_sum_squares, grad_neg_sum_squares
from constraints import *
from penalties import *
import time
import scipy
import numpy as np
"""
Tests for Hazan's core algorithm.

@author: Bharath Ramsundar
@email: bharath.ramsundar@gmail.com

TODOs:
    -) Remove m, n from the test cases.
"""

def test1():
    """
    Test bounded trace solver on function f(x)  = -\sum_k x_kk^2
    defined above.

    Now do a dummy optimization problem. The
    problem we consider is

        max - sum_k x_k^2
        subject to
            Tr(X) = 1

    The optimal solution is -1/n, where
    n is the dimension.
    """
    eps = 1e-3
    N_iter = 50
    # dimension of square matrix X
    dims = [16]
    for dim in dims:
        print("dim = %d" % dim)
        b = BoundedTraceSolver(neg_sum_squares, grad_neg_sum_squares, dim)
        X = b.solve(N_iter)
        fX = neg_sum_squares(X)
        print("\tTr(X) = %f" % np.trace(X))
        print("\tf(X) = %f" % fX)
        print("\tf* = %f" % (-1./dim))
        print("\t|f(X) - f*| = %f" % (np.abs(fX - (-1./dim))))
        assert np.abs(fX - (-1./dim)) < eps

def test2a():
    """
    Check equality constraints for log_sum_exp constraints
    """
    eps = 1e-3
    N_iter = 50
    dim, As, bs, Cs, ds, Fs, gradFs, Gs, gradGs = \
           simple_equality_constraint()
    M = compute_scale(len(As), len(Cs), len(Fs), len(Gs), eps)
    def f(X):
        return log_sum_exp_penalty(X, M, As, bs, Cs, ds, Fs, Gs)
    def gradf(X):
        return log_sum_exp_grad_penalty(X, M, As,
                    bs, Cs, ds, Fs, gradFs, Gs, gradGs)
    B = BoundedTraceSolver(f, gradf, dim)
    X, elapsed  = run_experiment(B, N_iter)
    succeed = not (f(X) < -eps)
    print "\tComputation Time (s): ", elapsed
    assert succeed == True

#def test2b():
#    """
#    BROKEN: Check equality constraints for neg_max constraints
#    TODO: Fix this test
#    """
#    N_iter = 50
#    assert True == False
#    m, n, M, dim, eps, As, bs, Cs, ds, Fs, gradFs, Gs, gradGs = \
#           simple_equality_constraint(N_iter)
#    def f(X):
#        return neg_max_penalty(X, M, As, bs, Cs, ds, Fs, Gs)
#    def gradf(X):
#        return neg_max_grad_penalty(X, M, As, bs, Cs, ds, Fs, gradFs,
#                Gs, gradGs, eps)
#    run_experiment(f, gradf, dim, N_iter)

def test3a():
    """
    Check equality and inequality constraints for log_sum_exp penalty
    """
    eps = 1e-3
    N_iter = 50
    dim, As, bs, Cs, ds, Fs, gradFs, Gs, gradGs = \
            simple_equality_and_inequality_constraint()
    M = compute_scale(len(As), len(Cs), len(Fs), len(Gs), eps)
    def f(X):
        return log_sum_exp_penalty(X, M, As, bs, Cs, ds, Fs, Gs)
    def gradf(X):
        return log_sum_exp_grad_penalty(X, M, As, bs, Cs, ds,
                Fs, gradFs, Gs, gradGs)
    B = BoundedTraceSolver(f, gradf, dim)
    X, elapsed  = run_experiment(B, N_iter)
    succeed = not (f(X) < -eps)
    print "\tComputation Time (s): ", elapsed
    assert succeed == True

#def test3b():
#    """
#    BROKEN: Check equality and inequality constraints for neg_max penalty
#    TODO: Fix this test
#    """
#    assert True == False
#    N_iter = 50
#    m, n, M, As, bs, Cs, ds, dim, eps, Fs, gradFs, Gs, gradGs = \
#            simple_constraint(N_iter)
#    def f(X):
#        return neg_max_penalty(X, M, As, bs, Cs, ds, Fs, Gs)
#    def gradf(X):
#        return neg_max_grad_penalty(X, M, As, bs, Cs, ds, Fs, gradFs,
#                Gs, gradGs, eps)
#    X, fX, SUCCEED = run_experiment(f, gradf, dim, N_iter)

#def test3c():
#    """
#    BROKEN: Check equality and inequality constraints for neg_max penalty
#    with log_sum_exp gradients.
#    TODO: Fix this test
#    """
#    assert True == False
#    N_iter = 50
#    m, n, M, As, bs, Cs, ds, dim, eps, Fs, gradFs, Gs, gradGs = \
#            simple_constraint(N_iter)
#    def f(X):
#        return neg_max_penalty(X, M, As, bs, Cs, ds, Fs, Gs)
#    def gradf(X):
#        return log_sum_exp_grad_penalty(X, M,
#                As, bs, Cs, ds, Fs, gradFs, Gs, gradGs, eps)
#    X, fX, SUCCEED = run_experiment(f, gradf, dim, N_iter)

#def test4a():
#    """
#    BROKEN: Check quadratic inequality constraints for neg_max penalty
#    and gradients.
#    TODO: Fix this test
#    """
#    assert True == False
#    N_iter = 50
#    dim, M, As, bs, Cs, ds, Fs, gradFs, Gs, gradGs, eps = \
#            quadratic_inequality(N_iter)
#    def f(X):
#        return neg_max_penalty(X, M, As, bs, Cs, ds, Fs, Gs)
#    def gradf(X):
#        return neg_max_grad_penalty(X, M,
#                As, bs, Cs, ds, Fs, gradFs, Gs, gradGs, eps)
#    X, fX, SUCCEED = run_experiment(f, gradf, dim, N_iter)

def test4b():
    """
    Check quadratic inequality for log_sum_exp penalty and gradients.
    """
    eps = 1e-3
    N_iter = 50
    dim, As, bs, Cs, ds, Fs, gradFs, Gs, gradGs = \
            quadratic_inequality()
    M = compute_scale(len(As), len(Cs), len(Fs), len(Gs), eps)
    def f(X):
        return log_sum_exp_penalty(X, M, As, bs, Cs, ds, Fs, Gs)
    def gradf(X):
        return log_sum_exp_grad_penalty(X, M,
                As, bs, Cs, ds, Fs, gradFs, Gs, gradGs)
    B = BoundedTraceSolver(f, gradf, dim)
    X, elapsed  = run_experiment(B, N_iter)
    succeed = not (f(X) < -eps)
    print "\tComputation Time (s): ", elapsed
    assert succeed == True

#def test5a():
#    """
#    BROKEN: quadratic equality constraints for neg_max penalty
#    and gradients.
#    """
#    assert True == False
#    N_iter = 50
#    dim, M, As, bs, Cs, ds, Fs, gradFs, Gs, gradGs, eps = \
#            quadratic_equality(N_iter)
#    def f(X):
#        return neg_max_penalty(X, M, As, bs, Cs, ds, Fs, Gs)
#    def gradf(X):
#        return neg_max_grad_penalty(X, M,
#                As, bs, Cs, ds, Fs, gradFs, Gs, gradGs, eps)
#    X, fX, SUCCEED = run_experiment(f, gradf, dim, N_iter)

def test6a():
    """
    Stress test inequality constraints for log_sum_exp penalty.
    """
    eps = 1e-3
    dims = [4,16]
    N_iter = 50
    for dim in dims:
        As, bs, Cs, ds, Fs, gradFs, Gs, gradGs = \
                stress_inequalities(dim)
        M = compute_scale(len(As), len(Cs), len(Fs), len(Gs), eps)
        def gen_f(M, As, bs, Cs, ds, Fs, Gs):
            def f(X):
                return log_sum_exp_penalty(X, M, As, bs, Cs, ds, Fs, Gs)
            return f
        f = gen_f(M, As, bs, Cs, ds, Fs, Gs)
        def gen_gradf(M, As, bs, Cs, ds, Fs, gradFs, Gs, gradGs):
            def gradf(X):
                return log_sum_exp_grad_penalty(X, M,
                        As, bs, Cs, ds, Fs, gradFs, Gs, gradGs)
            return gradf
        gradf = gen_gradf(M, As, bs, Cs, ds, Fs, gradFs, Gs, gradGs)
        B = BoundedTraceSolver(f, gradf, dim)
        X, elapsed  = run_experiment(B, N_iter)
        succeed = not (f(X) < -eps)
        print "\tComputation Time (s): ", elapsed
        assert succeed == True

#def test6b():
#    """
#    BROKEN: Stress test inequality constraints for neg_max_sum penatly
#    and log_sum_exp gradient.
#    """
#    assert True == False
#    dims = [4,16]
#    N_iter = 50
#    for dim in dims:
#        m, n, M, As, bs, Cs, ds, eps, Fs, gradFs, Gs, gradGs = \
#                stress_inequalities(dim, N_iter)
#        def f(X):
#            return neg_max_penalty(X, M, As, bs, Cs, ds, Fs, Gs)
#        def gradf(X):
#            return log_sum_exp_grad_penalty(X, M,
#                    As, bs, Cs, ds, Fs, gradFs, Gs, gradGs, eps)
#        run_experiment(f, gradf, dim, N_iter)

#def test6c():
#    """
#    BROKEN: Stress test inequality constraints for neg_max_penalty
#    """
#    assert True == False
#    dims = [4,16]
#    N_iter = 50
#    for dim in dims:
#        m, n, M, As, bs, Cs, ds, eps, Fs, gradFs, Gs, gradGs = \
#                stress_inequalities(dim, N_iter)
#        def f(X):
#            return neg_max_penalty(X, M, As, bs, Cs, ds, Fs, Gs)
#        def gradf(X):
#            return neg_max_grad_penalty(X, M, As, bs, Cs, ds,
#                    Fs, gradFs, Gs, gradGs, eps)
#        run_experiment(f, gradf, dim, N_iter)

def test7a():
    """
    Stress test equality constraints for log_sum_exp_penalty
    """
    eps = 1e-3
    dims = [4,16]
    N_iter = 50
    for dim in dims:
        As, bs, Cs, ds, Fs, gradFs, Gs, gradGs = \
                stress_equalities(dim)
        M = compute_scale(len(As), len(Cs), len(Fs), len(Gs), eps)
        def gen_f(M, As, bs, Cs, ds, Fs, Gs):
            def f(X):
                return log_sum_exp_penalty(X, M, As, bs, Cs, ds, Fs, Gs)
            return f
        f = gen_f(M, As, bs, Cs, ds, Fs, Gs)
        def gen_gradf(M, As, bs, Cs, ds, Fs, gradFs, Gs, gradGs):
            def gradf(X):
                return log_sum_exp_grad_penalty(X, M,
                        As, bs, Cs, ds, Fs, gradFs, Gs, gradGs)
            return gradf
        gradf = gen_gradf(M, As, bs, Cs, ds, Fs, gradFs, Gs, gradGs)
        B = BoundedTraceSolver(f, gradf, dim)
        X, elapsed  = run_experiment(B, N_iter)
        succeed = not (f(X) < -eps)
        print "\tComputation Time (s): ", elapsed
        assert succeed == True

#def test7b():
#    """
#    BROKEN: Stress test equality constraints for neg_max_penalty
#    """
#    assert True == False
#    dims = [4,16]
#    N_iter = 50
#    for dim in dims:
#        m, n, M, As, bs, Cs, ds, eps, Fs, gradFs, Gs, gradGs = \
#                stress_equalities(dim, N_iter)
#        def f(X):
#            return neg_max_penalty(X, M, As, bs, Cs, ds, Fs, Gs)
#        def gradf(X):
#            return log_sum_exp_grad_penalty(X, M, As, bs, Cs, ds,
#                    Fs, gradFs, Gs, gradGs, eps)
#        run_experiment(f, gradf, dim, N_iter)

#def test7c():
#    """
#    BROKEN: Stress test equality constraints for neg_max_penalty
#    """
#    assert True == False
#    dims = [4,16]
#    N_iter = 50
#    for dim in dims:
#        m, n, M, As, bs, Cs, ds, eps, Fs, gradFs, Gs, gradGs = \
#                stress_equalities(dim, N_iter)
#        def f(X):
#            return neg_max_penalty(X, M, As, bs, Cs, ds, Fs, Gs)
#        def gradf(X):
#            return neg_max_grad_penalty(X, M, As, bs, Cs, ds,
#                    Fs, gradFs, Gs, gradGs, eps)
#        run_experiment(f, gradf, dim, N_iter)

def test8a():
    """
    Stress test equality and inequality constraints for log_sum_exp_penalty
    """
    eps = 1e-3
    dims = [4,16]
    N_iter = 200
    for dim in dims:
        As, bs, Cs, ds, Fs, gradFs, Gs, gradGs = \
               stress_inequalities_and_equalities(dim)
        M = compute_scale(len(As), len(Cs), len(Fs), len(Gs), eps)
        def gen_f(M, As, bs, Cs, ds, Fs, Gs):
            def f(X):
                return log_sum_exp_penalty(X, M, As, bs, Cs, ds, Fs, Gs)
            return f
        f = gen_f(M, As, bs, Cs, ds, Fs, Gs)
        def gen_gradf(M, As, bs, Cs, ds, Fs, gradFs, Gs, gradGs):
            def gradf(X):
                return log_sum_exp_grad_penalty(X, M,
                        As, bs, Cs, ds, Fs, gradFs, Gs, gradGs)
            return gradf
        gradf = gen_gradf(M, As, bs, Cs, ds, Fs, gradFs, Gs, gradGs)
        B = BoundedTraceSolver(f, gradf, dim)
        X, elapsed  = run_experiment(B, N_iter)
        succeed = not (f(X) < -eps)
        print "\tComputation Time (s): ", elapsed
        assert succeed == True

#def test8b():
#    """
#    BROKEN Stress test equality constraints for neg_max_penalty
#    """
#    assert True == False
#    dims = [4, 16]
#    N_iter = 50
#    for dim in dims:
#        m, n, M, As, bs, Cs, ds, eps, Fs, gradFs, Gs, gradGs = \
#               stress_inequalities_and_equalities(dim, N_iter)
#        def f(X):
#            return neg_max_penalty(X, M, As, bs, Cs, ds, Fs, Gs)
#        def gradf(X):
#            return log_sum_exp_grad_penalty(X, M, As, bs, Cs, ds,
#                    Fs, gradFs, Gs, gradGs, eps)
#        X, fX, SUCCEED = run_experiment(f, gradf, dim, N_iter)

#def test8c():
#    """
#    BROKEN Stress test equality constraints for neg_max_penalty
#    """
#    assert True == False
#    dims = [4, 16]
#    N_iter = 50
#    for dim in dims:
#        m, n, M, As, bs, Cs, ds, eps, Fs, gradFs, Gs, gradGs = \
#               stress_inequalities_and_equalities(dim, N_iter)
#        def f(X):
#            return neg_max_penalty(X, M, As, bs, Cs, ds, Fs, Gs)
#        def gradf(X):
#            return neg_max_grad_penalty(X, M,
#                        As, bs, Cs, ds, Fs, gradFs, Gs, gradGs, eps)
#        X, fX, SUCCEED = run_experiment(f, gradf, dim, N_iter)

def test9a():
    """
    Test block equality constraints.
    """
    eps = 1e-3
    dims = [4]
    N_iter = 200
    for dim in dims:
        block_dim = int(dim/2)
        # Generate random configurations
        A = 0.25*np.eye(block_dim)
        B = np.eye(block_dim)
        D = np.eye(block_dim)
        tr_B_D = np.trace(B) + np.trace(D)
        B = B / tr_B_D
        D = D / tr_B_D
        As, bs, Cs, ds, Fs, gradFs, Gs, gradGs = \
                basic_batch_equality(dim, A, B, D)
        M = compute_scale(len(As), len(Cs), len(Fs), len(Gs), eps)
        def f(X):
            return log_sum_exp_penalty(X, M, As, bs, Cs, ds, Fs, Gs)
        def gradf(X):
            return log_sum_exp_grad_penalty(X, M,
                        As, bs, Cs, ds, Fs, gradFs, Gs, gradGs)
        B = BoundedTraceSolver(f, gradf, dim)
        X, elapsed  = run_experiment(B, N_iter)
        succeed = not (f(X) < -eps)
        print "\tComputation Time (s): ", elapsed
        assert succeed == True

def run_experiment(B, N_iter, disp=True, debug=False):
    start = time.clock()
    X = B.solve(N_iter, disp=disp, debug=debug)
    elapsed = (time.clock() - start)
    return X, elapsed
