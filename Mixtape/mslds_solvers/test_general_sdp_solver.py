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

def testSchurComplement():
    """
    Specifies a Schur complement convex program of form

    minimize -log det R
          --------------
         |D-ADA.T  I    |
    X =  |   I     R    |
         |            R |
          --------------
    X is PSD
    """
    dim = 1
    cdim = 3 * dim
    g = GeneralSDPHazanSolver()
    As, bs, Cs, ds, = [], [], [], []
    Fs, gradFs, Gs, gradGs = [], [], [], []

    # Generate initial data
    D = np.eye(dim)
    Dinv = np.linalg.inv(D)
    B = np.eye(dim)
    A = 0.5 * np.eye(dim)
    """ We neeed to enforce constant equalities in X.
      -------------
     |D-ADA.T   I    0 |
C =  | I        _    0 |
     | 0        0    _ |
      -------------
    """
    C = np.zeros((cdim, cdim))
    D_ADA_T_cds = (0, dim, 0, dim)
    D_ADA_T = D - np.dot(A, np.dot(D, A.T))
    I_1_cds = (0, dim, dim, 2*dim)
    I_2_cds = (dim, 2*dim, 0, dim)
    constraints = [(D_ADA_T_cds, D_ADA_T), (I_1_cds, np.eye(dim)),
            (I_2_cds, np.eye(dim))]
    # Add zero constraints
    constraints += [((2*dim, 3*dim, 0, 2*dim), np.zeros((dim, 2*dim))),
            ((0, 2*dim, 2*dim, 3*dim), np.zeros((2*dim, dim)))]
    def const_regions(X):
        return many_batch_equals(X, constraints)
    def grad_const_regions(X):
        return grad_many_batch_equals(X, constraints)
    Gs.append(const_regions)
    gradGs.append(grad_const_regions)

    """ We need to constraint linear inequalities
          ----------
         |          |
    C =  |     R    |
         |        R |
          ----------
    """
    R_cds = (2*dim, 3*dim, 2*dim, 3*dim)
    block_1_R_cds = (dim, 2*dim, dim, 2*dim)
    linear_constraints = [(1., R_cds, np.zeros((dim,dim)), block_1_R_cds)]

    def linear_regions(X):
        return many_batch_linear_equals(X, linear_constraints)
    def grad_linear_regions(X):
        return grad_many_batch_linear_equals(X, linear_constraints)
    Gs.append(linear_regions)
    gradGs.append(grad_linear_regions)

    # - log det R + Tr(RB)
    def h(X):
        R = get_entries(X, R_cds)
        return -np.log(np.linalg.det(R)) + np.trace(np.dot(R, B))
    # grad - log det R = -R^{-1} = -Q (see Boyd and Vandenberge, A4.1)
    # grad tr(RB) = B^T
    def gradh(X):
        grad = np.zeros(np.shape(X))
        R = get_entries(X, R_cds)
        Q = np.linalg.inv(R)
        gradR = -Q + B.T
        set_entries(grad, R_cds, gradR)
        set_entries(grad, block_1_R_cds, gradR)
        return grad

    R = 5
    L = -20
    U = 20
    eps = 3e-2
    N_iter = 100
    X_init = np.zeros((cdim, cdim))
    Q_init = 0.2 * np.eye(dim)
    R_init = np.linalg.inv(Q_init)
    set_entries(X_init, R_cds, R_init)
    set_entries(X_init, block_1_R_cds, R_init)
    import pdb
    pdb.set_trace()
    upper, lower, X_upper, X_lower, SUCCEED = g.solve(h, gradh, As, bs,
                Cs, ds, Fs, gradFs, Gs, gradGs, eps, cdim, R, U, L,
                N_iter, X_init=X_init)
    print "X_lower\n", X_lower
    if X_lower != None:
        print "h(X_lower)\n", h(X_lower)
    print "X_upper\n", X_upper
    if X_upper != None:
        print "h(X_upper)\n", h(X_upper)

def testA():
    """
    Specifies a simple version of the convex program required for
    A optimization.

    Given
        D, Q, C, B, E

    We want to solve for x = A_i. To do so, we solve

    max Tr [ Q^{-1} ([C - B] A.T + A [C - B].T + A E A.T]

          --------------------
         | D-Q    A           |
    X =  | A.T  D^{-1}        |
         |              I   A |
         |             A.T  I |
          --------------------
    X is PSD

    If A is dim by dim, then this matrix is 4 * dim by 4 * dim
    """
    dim = 1
    cdim = 4 * dim
    g = GeneralSDPHazanSolver()
    As = []
    bs = []
    Cs = []
    ds = []
    Fs = []
    gradFs = []
    Gs = []
    gradGs = []
    block_1_A_coords = (0, dim, dim, 2*dim)
    block_1_A_T_coords = (dim, 2*dim, 0, dim)
    block_2_A_coords = (2*dim, 3*dim, 3*dim, 4*dim)
    block_2_A_T_coords = (3*dim, 4*dim, 2*dim, 3*dim)

    # Generate random data
    D = np.random.rand(dim, dim)
    D = np.dot(D.T, D)
    Q = np.random.rand(dim, dim)
    Q = np.dot(Q.T, Q)


    # Tr [ Q^{-1} ([C - B] A.T + A [C - B].T + A E A.T]

    # Zero constraints
    Z_1_below = np.zeros((2*dim, 2*dim))
    def block_1_below_zeros(X):
        return batch_equals(X, Z_1_below, 2*dim, 4*dim, 0, 2*dim)
    def grad_block_1_below_zeros(X):
        return batch_equals_grad(X, Z_1_below, 2*dim, 4*dim, 0, 2*dim)
    Z_1_right = np.zeros((2*dim, 2*dim))
    def block_1_right_zeros(X):
        return batch_equals(X, Z_1_right, 0, 2*dim, 2*dim, 4*dim)
    def grad_block_1_right_zeros(X):
        return batch_equals(X, Z_1_right, 0, 2*dim, 2*dim, 4*dim)

    Gs += [block_1_below_zeros, block_1_right_zeros]
    gradGs += [grad_block_1_below_zeros, grad_block_1_right_zeros]


if __name__ == "__main__":
    #test1()
    #testQ()
    testSchurComplement()
    pass
