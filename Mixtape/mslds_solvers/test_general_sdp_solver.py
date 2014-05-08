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

def testQ():
    """
    Specifies the convex program required for Q optimization.

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

    """
    We need to enforce zero equalities in X.
      -------------
     | -        -    0 |
C =  | -        _    0 |
     | 0        0    _ |
      -------------
    """
    constraints += [((2*dim, 3*dim, 0, 2*dim), np.zeros((dim, 2*dim))),
            ((0, 2*dim, 2*dim, 3*dim), np.zeros((2*dim, dim)))]

    """
    We need to enforce constant equalities in X.
      -------------
     |D-ADA.T   I    _ |
C =  | I        _    _ |
     | _        _    _ |
      -------------
    """
    D_ADA_T_cds = (0, dim, 0, dim)
    D_ADA_T = D - np.dot(A, np.dot(D, A.T))
    I_1_cds = (0, dim, dim, 2*dim)
    I_2_cds = (dim, 2*dim, 0, dim)
    constraints = [(D_ADA_T_cds, D_ADA_T), (I_1_cds, np.eye(dim)),
            (I_2_cds, np.eye(dim))]

    # Add constraints to Gs
    def const_regions(X):
        return many_batch_equals(X, constraints)
    def grad_const_regions(X):
        return grad_many_batch_equals(X, constraints)
    Gs.append(const_regions)
    gradGs.append(grad_const_regions)


    """ We need to enforce linear inequalities
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
    set_entries(X_init, D_ADA_T_cds, D_ADA_T)
    set_entries(X_init, I_1_cds, np.eye(dim))
    set_entries(X_init, I_2_cds, np.eye(dim))
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
    As, bs, Cs, ds, = [], [], [], []
    Fs, gradFs, Gs, gradGs = [], [], [], []

    block_1_A_coords = (0, dim, dim, 2*dim)
    block_1_A_T_coords = (dim, 2*dim, 0, dim)
    block_2_A_coords = (2*dim, 3*dim, 3*dim, 4*dim)
    block_2_A_T_coords = (3*dim, 4*dim, 2*dim, 3*dim)

    # Generate random data
    D = np.eye(dim)
    Q = 0.5 * np.eye(dim)
    C = 2 * np.eye(dim)
    B = np.eye(dim)
    E = np.eye(dim)

    """
    We need to enforce zero equalities in X
      ----------------------
     | _        _    0   0 |
C =  | _        _    0   0 |
     | 0        0    _   _ |
     | 0        0    _   _ |
      ----------------------
    """
    constraints += [((2*dim, 4*dim, 0, 2*dim), np.zeros((2*dim, 2*dim))),
            ((0, 2*dim, 2*dim, 4*dim), np.zeros((2*dim, 2*dim)))]

    """
    We need to enforce constant equalities in X
      ---------------------
     |D-Q       _    _   _ |
C =  | _     D^{-1}  _   _ |
     | _        _    I   _ |
     | _        _    _   I |
      ---------------------
    """
    D_Q_cds = (0, dim, 0, dim)
    D_Q = D-Q
    Dinv_cds = (dim, 2*dim, dim, 2*dim)
    Dinv = np.linalg.inv(D)
    I_1_cds = (2*dim, 3*dim, 2*dim, 3*dim)
    I_2_cds = (3*dim, 4*dim, 3*dim, 4*dim)
    constraints = [(D_Q_cds, D_Q), (Dinv_cds, Dinv),
            (I_1_cds, np.eye(dim)), (I_2_cds, np.eye(dim))]

    # Add constraints to Gs
    def const_regions(X):
        return many_batch_equals(X, constraints)
    def grad_const_regions(X):
        return grad_many_batch_equals(X, constraints)
    Gs.append(const_regions)
    gradGs.append(grad_const_regions)

    """ We need to enforce linear inequalities

          --------------------
         |  _     A     _   _ |
    X =  | A.T    _     _   _ |
         |  _     _     _   A |
         |  _     _    A.T  _ |
          --------------------
    """
    A_1_cds = (0, dim, dim, 2*dim)
    A_2_cds = (2*dim, 3*dim, 3*dim, 4*dim)
    linear_constraints = [(1., A_1_cds, np.zeros((dim, dim)), A_2_cds)]

    def linear_regions(X):
        return many_batch_linear_equals(X, linear_constraints)
    def grad_linear_regions(X):
        return grad_many_batch_linear_equals(X, linear_constraints)
    Gs.append(linear_regions)
    gradGs.append(grad_linear_regions)

    # Tr [ Q^{-1} ([C - B] A.T + A [C - B].T + A E A.T]
    def h(X):
        A_1 = get_entries(X, A_1_cds)
        term1 = np.dot(C-B, A_1.T)
        term2 = term1.T
        term3 = np.dot(A_1, np.dot(E, A_1.T))
        term = np.dot(Qinv, term1+term2+term3)
        return np.trace(term)
    # grad Tr [Q^{-1} (C - B) A.T] = Q^{-1} (C - B)
    # grad Tr [Q^{-1} A [C - B].T] = Q^{-T} (C - B)
    # grad Tr [Q^{-1} A E A.T] = Q^{-T} A E.T + Q^{-1} A E
    def gradh(X):
        grad = np.zeros(np.shape(X))
        A_1 = get_entries(X, A_1_cds)
        grad_term1 = np.dot(Qinv, C-B)
        grad_term2 = np.dot(Qinv.T, C-B)
        grad_term3 = np.dot(Qinv.T, np.dot(A_1, E.T)) + \
                        np.dot(Qinv, np.dot(A_1, E))
        gradA = grad_term1 + grad_term2 + grad_term3
        set_entries(grad, A_1_cds, gradA)
        set_entries(grad, A_2_cds, gradA)
        set_entries(grad, A_T_1_cds, gradA.T)
        set_entries(grad, A_T_2_cds, gradA.T)
        return grad
    R = 5
    L = -20
    U = 20
    eps = 3e-2
    N_iter = 100
    X_init = np.zeros((cdim, cdim))
    set_entries(X_init, D_Q_cds, D_Q)
    set_entries(Dinv_cds, Dinv)
    set_entries(I_1_cds, np.eye(dim))
    set_entries(I_2_cds, np.eye(dim))
    A_init = np.eye(dim)
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


if __name__ == "__main__":
    #test1()
    #testQ()
    testSchurComplement()
    pass
