from __future__ import division, print_function, absolute_import
from ..bounded_trace_sdp_solver import BoundedTraceSolver
from ..objectives import neg_sum_squares, grad_neg_sum_squares
from ..constraints import *
from ..penalties import *
import time
import scipy
import numpy as np
from nose.plugins.attrib import attr

"""
Tests for Hazan's core algorithm.

@author: Bharath Ramsundar
@email: bharath.ramsundar@gmail.com

"""

@attr('broken')
def test1():
    # Test bounded trace solver on function f(x)  = -\sum_k x_kk^2
    # defined above.
    #
    # Now do a dummy optimization problem. The
    # problem we consider is
    #
    #     max - sum_k x_k^2
    #     subject to
    #         Tr(X) = 1
    #
    # The optimal solution is -1/n, where
    # n is the dimension.

    eps = 1e-3
    N_iter = 50
    # dimension of square matrix X
    dims = [16]
    for dim in dims:
        print(("dim = %d" % dim))
        b = BoundedTraceSolver(neg_sum_squares, grad_neg_sum_squares, dim)
        X = b.solve(N_iter, methods=['frank_wolfe'], disp=False)
        fX = neg_sum_squares(X)
        print(("\tTr(X) = %f" % np.trace(X)))
        print(("\tf(X) = %f" % fX))
        print(("\tf* = %f" % (-1./dim)))
        print(("\t|f(X) - f*| = %f" % (np.abs(fX - (-1./dim)))))
        assert np.abs(fX - (-1./dim)) < eps

def test2():
    # Check equality constraints for log_sum_exp constraints

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
    X, elapsed  = run_experiment(B, N_iter, ['frank_wolfe'], disp=False)
    succeed = not (f(X) < -eps)
    print("\tComputation Time (s): ", elapsed)
    assert succeed == True

def test3():
    # Check equality and inequality constraints for log_sum_exp penalty

    eps = 1e-3
    N_iter = 100
    dim, As, bs, Cs, ds, Fs, gradFs, Gs, gradGs = \
            simple_equality_and_inequality_constraint()
    M = compute_scale(len(As), len(Cs), len(Fs), len(Gs), eps)
    def f(X):
        return log_sum_exp_penalty(X, M, As, bs, Cs, ds, Fs, Gs)
    def gradf(X):
        return log_sum_exp_grad_penalty(X, M, As, bs, Cs, ds,
                Fs, gradFs, Gs, gradGs)
    B = BoundedTraceSolver(f, gradf, dim)
    X, elapsed  = run_experiment(B, N_iter, ['frank_wolfe'], disp=False)
    succeed = not (f(X) < -eps)
    print("\tComputation Time (s): ", elapsed)
    assert succeed == True

def test4():
    # Check quadratic inequality for log_sum_exp penalty and gradients.

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
    X, elapsed  = run_experiment(B, N_iter, ['frank_wolfe'], disp=False)
    succeed = not (f(X) < -eps)
    print("\tComputation Time (s): ", elapsed)
    assert succeed == True

def test5():
    # Stress test inequality constraints for log_sum_exp penalty.

    eps = 1e-3
    dims = [4,16]
    N_iter = 50
    for dim in dims:
        As, bs, Cs, ds, Fs, gradFs, Gs, gradGs = \
                stress_inequalities(dim)
        M = compute_scale(len(As), len(Cs), len(Fs), len(Gs), eps)
        def f(X):
            return log_sum_exp_penalty(X, M, As, bs, Cs, ds, Fs, Gs)
        def gradf(X):
            return log_sum_exp_grad_penalty(X, M,
                    As, bs, Cs, ds, Fs, gradFs, Gs, gradGs)
        B = BoundedTraceSolver(f, gradf, dim)
        X, elapsed  = run_experiment(B, N_iter, ['frank_wolfe'],
                disp=False)
        succeed = not (f(X) < -eps)
        print("\tComputation Time (s): ", elapsed)
        assert succeed == True

def test6():
    # Stress test equality constraints for log_sum_exp_penalty

    eps = 1e-3
    dims = [4,16]
    N_iter = 50
    for dim in dims:
        As, bs, Cs, ds, Fs, gradFs, Gs, gradGs = \
                stress_equalities(dim)
        M = compute_scale(len(As), len(Cs), len(Fs), len(Gs), eps)
        def f(X):
            return log_sum_exp_penalty(X, M, As, bs, Cs, ds, Fs, Gs)
        def gradf(X):
            return log_sum_exp_grad_penalty(X, M,
                    As, bs, Cs, ds, Fs, gradFs, Gs, gradGs)
        B = BoundedTraceSolver(f, gradf, dim)
        X, elapsed  = run_experiment(B, N_iter, ['frank_wolfe'],
                disp=False)
        succeed = not (f(X) < -eps)
        print("\tComputation Time (s): ", elapsed)
        assert succeed == True


@attr('broken')
def test7():
    # Stress test equality and inequality constraints for log_sum_exp_penalty

    eps = 1e-5
    tol = 1e-2
    dims = [4,16]
    N_iter = 300
    for dim in dims:
        As, bs, Cs, ds, Fs, gradFs, Gs, gradGs = \
               stress_inequalities_and_equalities(dim)
        M = compute_scale(len(As), len(Cs), len(Fs), len(Gs), eps)
        def f(X):
            return log_sum_exp_penalty(X, M, As, bs, Cs, ds, Fs, Gs)
        def gradf(X):
            return log_sum_exp_grad_penalty(X, M,
                    As, bs, Cs, ds, Fs, gradFs, Gs, gradGs)
        B = BoundedTraceSolver(f, gradf, dim)
        X, elapsed  = run_experiment(B, N_iter, ['frank_wolfe'],
                disp=False)
        succeed = not (f(X) < -tol)
        print("\tComputation Time (s): ", elapsed)
        assert succeed == True

def test8():
    # Test block equality constraints.

    eps = 1e-5
    tol = 1e-2
    #dims = [4,16]
    dims = [2]
    N_iter = 200
    for dim in dims:
        block_dim = int(dim/2)
        A = (1./dim)*np.eye(block_dim)
        B = np.eye(block_dim)
        D = np.eye(block_dim)
        tr_B_D = np.trace(B) + np.trace(D)
        B = B / tr_B_D
        D = D / tr_B_D
        soln = np.zeros((dim, dim))
        soln[:block_dim, :block_dim] = B
        soln[:block_dim, block_dim:] = A
        soln[block_dim:, :block_dim] = A.T
        soln[block_dim:, block_dim:] = D
        As, bs, Cs, ds, Fs, gradFs, Gs, gradGs = \
                basic_batch_equality(dim, A, B, D)
        M = compute_scale(len(As), len(Cs), len(Fs), len(Gs), eps)
        def f(X):
            return log_sum_exp_penalty(X, M, As, bs, Cs, ds, Fs, Gs)
        def gradf(X):
            return log_sum_exp_grad_penalty(X, M,
                        As, bs, Cs, ds, Fs, gradFs, Gs, gradGs)
        B = BoundedTraceSolver(f, gradf, dim)
        X, elapsed  = run_experiment(B, N_iter,
                methods=['frank_wolfe'],
                    disp=True)
        succeed = not (f(X) < -tol)
        print("\tsoln\n", soln)
        print("\tX\n", X)
        print("\tComputation Time (s): ", elapsed)
        assert succeed == True

def run_experiment(B, N_iter, methods=[], disp=True,
        debug=False, early_exit=True):
    start = time.clock()
    X = B.solve(N_iter, disp=disp, debug=debug, methods=methods,
            early_exit=early_exit)
    elapsed = (time.clock() - start)
    return X, elapsed
