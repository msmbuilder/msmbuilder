import sys
sys.path.append("..")
from general_sdp_solver import *
from objectives import trace_obj, grad_trace_obj
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
    eps = 1e-4
    tol = 1e-3
    search_tol = 1e-2
    N_iter = 50
    dim = 2
    Rs = [10]
    L, U = (-10, 10)
    for R in Rs:
        dim, As, bs, Cs, ds, Fs, gradFs, Gs, gradGs = \
                simple_equality_constraint()
        g = GeneralSolver(R, L, U, dim, eps)
        g.save_constraints(trace_obj, grad_trace_obj, As, bs, Cs, ds,
                Fs, gradFs, Gs, gradGs)
        (alpha, _, _, _, _, succeed) = g.solve(N_iter, tol,
                interactive=False)
        assert succeed == True
        assert np.abs(alpha - 0.75) < search_tol
