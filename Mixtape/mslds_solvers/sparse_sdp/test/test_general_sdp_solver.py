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

def testA():
    """
    Specifies a simple version of the convex program required for
    A optimization.

    min Tr [ Q^{-1} ([C - B] A.T + A [C - B].T + A E A.T]

          --------------------
         | D-Q    A           |
    X =  | A.T  D^{-1}        |
         |              I   A |
         |             A.T  I |
          --------------------
    X is PSD

    If A is dim by dim, then this matrix is 4 * dim by 4 * dim.
    The solution to this problem is A = 0 when dim = 1.
    """
    dim = 1
    cdim = 4 * dim
    g = GeneralSDPHazanSolver()


    # Generate random data
    D = np.eye(dim)
    Q = 0.5 * np.eye(dim)
    Qinv = np.linalg.inv(Q)
    C = 2 * np.eye(dim)
    B = np.eye(dim)
    E = np.eye(dim)

    R = 5
    L = 0
    U = 5
    eps = 1e-1
    N_iter = 100
    X_init = np.zeros((cdim, cdim))
    set_entries(X_init, D_Q_cds, D_Q)
    set_entries(X_init, Dinv_cds, Dinv)
    set_entries(X_init, I_1_cds, np.eye(dim))
    set_entries(X_init, I_2_cds, np.eye(dim))
    A_init = (1./np.sqrt(2)) * np.eye(dim)
    set_entries(X_init, A_1_cds, A_init)
    set_entries(X_init, A_2_cds, A_init)
    set_entries(X_init, A_T_1_cds, A_init.T)
    set_entries(X_init, A_T_2_cds, A_init.T)
    upper, lower, X_upper, X_lower, SUCCEED = g.solve(h, gradh, As, bs,
                Cs, ds, Fs, gradFs, Gs, gradGs, eps, cdim, R, U, L,
                N_iter, X_init=X_init)
    print "X_lower\n", X_lower
    if X_lower != None:
        print "h(X_lower)\n", h(X_lower)
    print "X_upper\n", X_upper
    if X_upper != None:
        print "h(X_upper)\n", h(X_upper)
