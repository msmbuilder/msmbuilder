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
    Specifies a Schur complement SDP of form

    minimize -log det Q^{-1}
          ------------
         |D-Q   A     |
    X =  |A.T D^{-1}  |
         |           Q|
          ------------
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
    A = 0.5 * np.eye(dim)
    """ We neeed to enforce constant equalities in X.
      -------------
     |D-_   A    0 |
C =  |A.T D^{-1} 0 |
     | 0    0    _ |
      -------------
    """
    C = np.zeros((cdim, cdim))
    A_1_cds = (0, dim, dim, 2*dim)
    A_T_1_cds = (dim, 2*dim, 0, dim)
    Dinv_1_cds  = (dim, 2*dim, dim, 2*dim)
    constraints = [(A_1_cds, A), (A_T_1_cds, A.T),
            (Dinv_1_cds, Dinv)]
    def const_regions(X):
        return many_batch_equals(X, constraints)
    def grad_const_regions(X):
        return grad_many_batch_equals(X, constraints)
    Gs.append(const_regions)
    gradGs.append(grad_const_regions)

    """ We need to constraint linear inequalities
          -----------
         |D-Q        |
    C =  |           |
         |         Q |
          -----------
    """
    Q_cds = (2*dim, 3*dim, 2*dim, 3*dim)
    block_1_Q_cds = (0, dim, 0, dim)
    linear_constraints = [(-1., Q_cds, D, block_1_Q_cds)]

    def linear_regions(X):
        return many_batch_linear_equals(X, linear_constraints)
    def grad_linear_regions(X):
        return grad_many_batch_linear_equals(X, linear_constraints)
    Gs.append(linear_regions)
    gradGs.append(grad_linear_regions)

    # log det Q^{-1} = - log det Q
    def h(X):
        Q = get_entries(X, Q_cds)
        #D_Q = get_entries(X, block_1_Q_cds)
        #block_1_Q = -D_Q + D
        #return -np.log(np.linalg.det(Q)) - np.log(np.linalg.det(block_1_Q))
        return np.log(np.linalg.det(Q))
    # grad log det Q^{-1} = -Q^{-1} (see Boyd and Vandenberge, A4.1)
    def gradh(X):
        grad = np.zeros(np.shape(X))
        Q = get_entries(X, Q_cds)
        # Look into avoiding this computation if possible
        eigs_Q = np.linalg.eigh(Q)[0]
        max_eig_Q = np.amax(eigs_Q)
        max_eig_Qinv = 1./max_eig_Q
        Qinv = np.linalg.inv(Q)
        # Scale Qinv down to unit spectral norm
        Qinv = (1/max_eig_Qinv) * Qinv
        # Scale Qinv to have spectral norm the same as Q
        Qinv *= max_eig_Q
        gradQ = Qinv
        set_entries(grad, Q_cds, gradQ)
        set_entries(grad, block_1_Q_cds, gradQ)
        return grad

    D_upper = np.trace(D)
    D_inv_upper = np.trace(Dinv)
    R = (D_upper + D_inv_upper + D_upper)
    L = 0
    U = 25
    eps = 3e-2
    N_iter = 150
    X_init = np.zeros((cdim, cdim))
    Q_init = 0.2 * np.eye(dim)
    set_entries(X_init, Q_cds, Q_init)
    set_entries(X_init, block_1_Q_cds, D - Q_init)
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


def testQ():
    """
    Specifies a simple version of the convex program required for
    Q optimization

    Given A, F = B^{.5},

    We want to solve for x = [s vec(Z) vec(Q)]. To do so, we
    construct and solve the following optimization problem

    max Tr Z + log det Q^{-1}

          -----------------------
         | Z   F                 |
         |F.T  Q                 |
         |        D-Q   A        |
    X =  |        A.T D^{-1}     |
         |                   Q   |
         |                     Z |
          -----------------------
    X is PSD

    If Q is dim by dim, then this matrix is
    6 * dim by 6 * dim

    TODO: Think of how to shrink this representation
    Ideas:
        1) Add specific zero penalty operations so
           we don't need to do python for-loops. ====> Done
        2) Extend Hazan's method to simultaneously maintain multiple
           PSD matrices instead of just one.
    """
    dim = 1
    cdim = 6 * dim
    g = GeneralSDPHazanSolver()
    As, bs, Cs, ds, = [], [], [], []
    Fs, gradFs, Gs, gradGs = [], [], [], []

    # Generate initial data
    B = 1 * np.eye(dim)
    D = np.eye(dim)
    Dinv = np.linalg.inv(D)
    F = scipy.linalg.sqrtm(B)
    A = np.eye(dim)

    """ We neeed to enforce constant equalities in X.
      ------------------------
     | _   F   0    0    0  0 |
     |F.T  _   0    0    0  0 |
     | 0   0  D-_   A    0  0 |
C =  | 0   0  A.T D^{-1} 0  0 |
     | 0   0   0    0    _  0 |
     | 0   0   0    0    0  _ |
      ------------------------
    """
    C = np.zeros((cdim, cdim))
    F_1_cds = (0, dim, dim, 2*dim)
    F_T_1_cds = (dim, 2*dim, 0, dim)

    A_2_cds = (2*dim, 3*dim, 3*dim, 4*dim)
    A_T_2_cds = (3*dim, 4*dim, 2*dim, 3*dim)

    Dinv_2_cds = (3*dim, 4*dim, 3*dim, 4*dim)

    Z_1_below =  np.zeros((4*dim, 2*dim))
    Z_1_below_cds = (2*dim, 6*dim, 0, 2*dim)

    Z_1_right =  np.zeros((2*dim, 4*dim))
    Z_1_right_cds = (0, 2*dim, 2, 6*dim)

    Z_2_below = np.zeros((2*dim, 2*dim))
    Z_2_below_cds = (4*dim, 6*dim, 2*dim, 4*dim)

    Z_2_right = np.zeros((2*dim, 2*dim))
    Z_2_right_cds = (2*dim, 4*dim, 4*dim, 6*dim)

    Z_3_below = np.zeros((dim, dim))
    Z_3_below_cds = (5*dim, 6*dim, 4*dim, 5*dim)

    Z_3_right = np.zeros((dim, dim))
    Z_3_right_cds = (4*dim, 5*dim, 5*dim, 6*dim)

    constraints = [(F_1_cds, F), (F_T_1_cds, F.T), (A_2_cds, A),
        (A_T_2_cds, A.T), (Dinv_2_cds, Dinv),
        (Z_1_below_cds, Z_1_below), (Z_1_right_cds, Z_1_right),
        (Z_2_below_cds, Z_2_below), (Z_2_right_cds, Z_2_right),
        (Z_3_below_cds, Z_3_below), (Z_3_right_cds, Z_3_right)]
    def const_regions(X):
        return many_batch_equals(X, constraints)
    def grad_const_regions(X):
        return grad_many_batch_equals(X, constraints)
    Gs.append(const_regions)
    gradGs.append(grad_const_regions)

    """ We need to constraint linear inequalities
          -----------------------
         | Z                     |
         |     Q                 |
         |        D-Q            |
    C =  |                       |
         |                   Q   |
         |                     Z |
          -----------------------
    """

    Z_cds = (5*dim, 6*dim, 5*dim, 6*dim)
    Q_cds = (4*dim, 5*dim, 4*dim, 5*dim)
    block_1_Z_cds = (0, dim, 0, dim)
    block_1_Q_cds = (dim, 2*dim, dim, 2*dim)
    block_2_Q_cds = (2*dim, 3*dim, 2*dim, 3*dim)

    linear_constraints = \
            [( 1., Q_cds, np.zeros((dim, dim)), block_1_Q_cds),
             (-1., Q_cds, D, block_2_Q_cds),
             ( 1., Z_cds, np.zeros((dim, dim)), block_1_Z_cds)]

    def linear_regions(X):
        return many_batch_linear_equals(X, linear_constraints)
    def grad_linear_regions(X):
        return grad_many_batch_linear_equals(X, linear_constraints)
    Gs.append(linear_regions)
    gradGs.append(grad_linear_regions)

    # Tr Z + log det Q^{-1}
    def h(X):
        Z = get_entries(X, Z_cds)
        Q = get_entries(X, Q_cds)
        return np.trace(Z) - np.log(np.linalg.det(Q))
    # grad log det Q^{-1} = Q (see Boyd and Vandenberge, A4.1)
    # grad h = I_Z + Q
    def gradh(X):
        grad = np.zeros(np.shape(X))
        gradQ = get_entries(X, Q_cds)
        gradZ = np.eye(dim)
        set_entries(X, Z_cds, gradZ)
        set_entries(X, block_1_Z_cds, gradZ)
        set_entries(grad, Q_cds, gradQ)
        set_entries(grad, block_1_Q_cds, gradQ)
        set_entries(grad, block_2_Q_cds, gradQ)
        return grad

    N_iter = 100
    eps = 1.e-2
    D_upper = np.trace(D)
    D_inv_upper = np.trace(Dinv)
    Z_upper = np.trace(np.dot(F, np.dot(Dinv, F.T)))
    R = (Z_upper + D_upper + D_upper + D_inv_upper + D_upper + Z_upper)
    L = 0
    U = 2 * Z_upper
    print "R: ", R

    X_init = np.zeros((cdim, cdim))
    Q_init = 0.2 * np.eye(dim)
    set_entries(X_init, Q_cds, Q_init)
    set_entries(X_init, block_1_Q_cds, Q_init)
    set_entries(X_init, block_2_Q_cds, D - Q_init)
    import pdb
    pdb.set_trace()
    upper, lower, X_upper, X_lower, SUCCEED = g.solve(h, gradh, As, bs,
                Cs, ds, Fs, gradFs, Gs, gradGs, eps, cdim, R, U, L,
                N_iter, X_init=X_init)

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
