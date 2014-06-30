import sys
sys.path.append("..")
from general_sdp_solver import *
from objectives import *
from constraints import *
import scipy
import numpy as np

# Do a simple test of General SDP Solver with binary search

def test1():
    """
    A simple semidefinite program

    min Tr(X)
    subject to
        x_11 + 2 x_22 == 1
        Tr(X) = x_11 + x_22 <= 10
        X semidefinite

    The solution to this problem is

        X = [[0, 0],
             [0, .75]]

    from Lagrange multiplier.
    """
    tol = 1e-3
    search_tol = 1e-2
    N_iter = 50
    Rs = [10, 100, 1000]
    dim, As, bs, Cs, ds, Fs, gradFs, Gs, gradGs = \
            simple_equality_constraint()
    g = GeneralSolver()
    g.save_constraints(dim, trace_obj, grad_trace_obj, As, bs, Cs, ds,
            Fs, gradFs, Gs, gradGs)
    (U, X, succeed) = g.solve(N_iter, tol, search_tol, verbose=False,
            interactive=False, disp=True, debug=False, Rs = Rs)
    print "X:\n", X
    assert succeed == True
    assert np.abs(X[1,1] - 0.75) < search_tol

def test2():
    """
    A simple semidefinite program to test trace search
    min Tr(X)
    subject to
        x_11 + 2 x_22 == 50
        X semidefinite

    The solution to this problem is

        X = [[0, 0],
             [0, 25]]
    """
    tol = 1e-1
    search_tol = 1e-1
    N_iter = 50
    Rs = [10, 100]
    dim = 2
    As, bs = [], []
    Cs = [np.array([[ 1.,  0.],
                    [ 0.,  2.]])]
    ds = [50]
    Fs, gradFs, Gs, gradGs = [], [], [], []
    g = GeneralSolver()
    g.save_constraints(dim, trace_obj, grad_trace_obj, As, bs, Cs, ds,
            Fs, gradFs, Gs, gradGs)
    (U, X, succeed) = g.solve(N_iter, tol, search_tol, verbose=True,
            interactive=False, debug=False, Rs = Rs)
    print "X:\n", X
    assert succeed == True
    assert np.abs(np.trace(X) - 25) < 2 

def test3():
    """
    A simple quadratic program
    min x_1
    subject to
        x_1^2 + x_2^2 = 1

    The solution to this problem is

        X = [[ 0, 0],
             [ 0, 1]]
        X semidefinite
    """
    tol = 1e-2
    search_tol = 3e-2 # Figure out how to reduce this...
    N_iter = 50
    Rs = [10, 100]
    dim = 2
    As, bs, Cs, ds, Fs, gradFs = [], [], [], [], [], []
    g = lambda(X): X[0,0]**2 + X[1,1]**2 - 1.
    def gradg(X):
        (dim, _) = np.shape(X)
        grad = np.zeros(np.shape(X))
        grad[range(dim), range(dim)] = 2*X[range(dim), range(dim)]
        return grad
    Gs, gradGs = [g], [gradg]
    def obj(X):
        return X[0,0]
    def grad_obj(X):
        G = np.zeros(np.shape(X))
        G[0,0] = 1.
        return G
    g = GeneralSolver()
    g.save_constraints(dim, obj, grad_obj, As, bs, Cs, ds,
            Fs, gradFs, Gs, gradGs)
    (U, X, succeed) = g.solve(N_iter, tol, search_tol, verbose=False,
            interactive=False, debug=False, Rs = Rs)
    print "X:\n", X
    assert succeed == True
    assert np.abs(X[0,0] - 0) < search_tol
