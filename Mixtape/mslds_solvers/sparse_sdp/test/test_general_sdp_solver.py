import sys
sys.path.append("..")
from hazan import *
from hazan_penalties import *
import scipy
import numpy as np

# Do a simple test of General SDP Solver with binary search

def test1():
    """
    A simple semidefinite program

    max x_11 + x_22
    subject to
        x_11 + 2 x_22 == 1
        x_12 == x_21 == 0
        Tr(X) = x_11 + x_22 <= 1
        X semidefinite

    The solution to this problem is

        X = [[1, 0],
             [0, 0]]

    from Lagrange multiplier.
    """
    dim = 2
    N_iter = 400
    eps = 1e-2
    g = GeneralSDPHazanSolver()
    def h(X):
        return X[0,0] + X[1,1]
    def gradh(X):
        return np.eye(2)
    As = []
    bs = []
    Cs = [np.array([[1., 0.],
                    [0., 2.]]),
          np.array([[0., 1.],
                    [0., 0.]]),
          np.array([[0., 0.],
                    [1., 0.]])]
    ds = [1., 0., 0.]
    Fs = []
    gradFs = []
    Gs = []
    gradGs = []
    U = 2
    L = 0
    R = 1.
    upper, lower, X_upper, X_lower, SUCCEED = g.solve(h, gradh, As, bs,
                Cs, ds, Fs, gradFs, Gs, gradGs, eps, dim, R, U, L, N_iter)
    print
    print "General SDP Solver Finished"
    print "SUCCEED: ", SUCCEED
    print "upper: ", upper
    print "X_upper:\n", X_upper
    print "X_lower:\n", X_lower
    print "lower: ", lower
