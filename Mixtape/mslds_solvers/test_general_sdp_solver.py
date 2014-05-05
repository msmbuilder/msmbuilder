from hazan import *
from hazan_penalties import batch_equals, batch_linear_equals
from hazan_penalties import batch_equals_grad, grad_batch_linear_equals
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
    As = []
    bs = []
    Cs = []
    ds = []
    Fs = []
    gradFs = []
    Gs = []
    gradGs = []
    Z_coords = (5*dim, 6*dim, 5*dim, 6*dim)
    Q_coords = (4*dim, 5*dim, 4*dim, 5*dim)
    block_1_Z_coords = (0, dim, 0, dim)
    block_1_Q_coords = (dim, 2*dim, dim, 2*dim)
    block_2_Q_coords = (2*dim, 3*dim, 2*dim, 3*dim)
    # Generate random initial data
    B = np.random.rand(dim, dim)
    B = np.dot(B.T, B)
    D = np.random.rand(dim, dim)
    D = np.dot(D.T, D)
    Dinv = np.linalg.inv(D)
    F = scipy.linalg.sqrtm(B)
    A = np.random.rand(dim, dim)
    # Tr Z + log det Q^{-1}
    def h(X):
        z_x_low, z_x_hi, z_y_low, z_y_hi = Z_coords
        q_x_low, q_x_hi, q_y_low, q_y_hi = Q_coords
        Z = X[z_x_low:z_x_hi, z_y_low:z_y_hi]
        Q = X[q_x_low:q_x_hi, q_y_low:q_y_hi]
        return np.trace(Z) - np.log(np.linalg.det(Q))
    # grad log det Q^{-1} = Q (see Boyd and Vandenberge, A4.1)
    # grad h = I_Z + Q
    def gradh(X):
        z_x_low, z_x_hi, z_y_low, z_y_hi = Z_coords
        q_x_low, q_x_hi, q_y_low, q_y_hi = Q_coords
        grad = np.zeros(np.shape(X))
        Z = X[z_x_low:z_x_hi, z_y_low:z_y_hi]
        grad[z_x_low:z_x_hi, z_y_low:z_y_hi] = np.eye(dim)
        Q = X[q_x_low:q_x_hi, q_y_low:q_y_hi]
        grad[q_x_low:q_x_hi, q_y_low:q_y_hi] = Q
        return grad

    #  -------
    # | Z   F |
    # |F.T  Q |
    #  -------
    def block_1_F(X):
        return batch_equals(X, F, 0, dim, dim, 2*dim)
    def grad_block_1_F(X):
        return batch_equals_grad(X, F, 0, dim, dim, 2*dim)
    def block_1_F_T(X):
        return batch_equals(X, F.T, dim, 2*dim, 0, dim)
    def grad_block_1_F_T(X):
        return batch_equals_grad(X, F.T, dim, 2*dim, 0, dim)
    Gs += [block_1_F, block_1_F_T]
    gradGs += [grad_block_1_F, grad_block_1_F_T]

    #  -----------
    # | D-Q   A   |
    # | A.T D^{-1}|
    #  -----------
    def block_2_A(X):
        return batch_equals(X, A, 0, dim, dim, 2*dim)
    def grad_block_2_A(X):
        return batch_equals_grad(X, A, 0, dim, dim, 2*dim)
    def block_2_A_T(X):
        return batch_equals(X, A.T, dim, 2*dim, 0, dim)
    def grad_block_2_A_T(X):
        return grad_batch_equals(X, A.T, dim, 2*dim, 0, dim)
    def block_2_Dinv(X):
        return batch_equals(X, Dinv, 3*dim, 4*dim, 3*dim, 4*dim)
    def grad_block_2_Dinv(X):
        return batch_equals(X, Dinv, 3*dim, 4*dim, 3*dim, 4*dim)
    Gs += [block_2_A, block_2_A_T, block_2_Dinv]
    gradGs += [grad_block_2_A, grad_block_2_A_T, grad_block_2_Dinv]
    # ---
    #| Q |
    # ---
    c = 1.
    Z = np.zeros((dim, dim))
    def block_1_Q(X):
        return batch_linear_equals(X, c, Q_coords, Z, block_1_Q_coords)
    def grad_block_1_Q(X):
        return grad_batch_linear_equals(X, c, Q_coords, Z,
                grad_block_1_Q_coords)
    d = -1.
    def block_2_Q(X):
        return batch_linear_equals(X, d, Q_coords, D, block_2_Q_coords)
    def grad_block_2_Q(X):
        return grad_batch_linear_equals(X, d, Q_coords, D,
                block_2_Q_coords)
    Gs += [block_1_Q, block_2_Q]
    gradGs += [grad_block_1_Q, grad_block_2_Q]
    # ---
    #| Z |
    # ---
    c = 1.
    Z = np.zeros((dim, dim))
    def block_1_Z(X):
        return batch_linear_equals(X, c, Z_coords, Z, block_1_Z_coords)
    def grad_block_1_Z(X):
        return grad_batch_linear_equals(X, c, Z_coords, Z,
                block_1_Z_coords)
    Gs += [block_1_Z]
    gradGs += [grad_block_1_Z]

    # Zero constraints
    Z_1_below = np.zeros((4*dim, 2*dim))
    def block_1_below_zeros(X):
        return batch_equals(X, Z_1_below, 2*dim, 6*dim, 0, 2*dim)
    def grad_block_1_below_zeros(X):
        return batch_equals_grad(X, Z_1_below, 2*dim, 6*dim, 0, 2*dim)
    Z_1_right = np.zeros((2*dim, 4*dim))
    def block_1_right_zeros(X):
        return batch_equals(X, Z_1_right, 0, 2*dim, 2*dim, 6*dim)
    def grad_block_1_right_zeros(X):
        return batch_equals(X, Z_1_right, 0, 2*dim, 2*dim, 6*dim)

    Gs += [block_1_below_zeros, block_1_right_zeros]
    gradGs += [grad_block_1_below_zeros, grad_block_1_right_zeros]

    R = 10
    U = 10
    L = 0
    N_iter = 10
    eps = 1.e-2
    #import pdb
    #pdb.set_trace()
    D_upper = np.trace(D)
    Q_upper = np.trace(D)
    Q_inv_upper = np.trace(Dinv)
    D_inv_upper = np.trace(Dinv)
    Z_upper = np.trace(np.dot(F, np.dot(Dinv, F.T)))
    R = (Z_upper + Q_upper + D_upper + D_inv_upper + Q_upper + Z_upper)

    X_init = np.zeros((cdim, cdim))
    Q_init = np.eye(dim)
    set_entries(X_init, Q_coords, Q_init)
    set_entries(X_init, block_1_Q_coords, Q_init)
    set_entries(X_init, block_2_Q_coords, Q_init)
    upper, lower, X_upper, X_lower, SUCCEED = g.solve(h, gradh, As, bs,
                Cs, ds, Fs, gradFs, Gs, gradGs, eps, cdim, R, U, L,
                N_iter, X_init=X_init)

def set_entries(X, coords, Z):
    x_low, x_hi, y_low, y_hi = coords
    X[x_low:x_hi, y_low:y_hi] = Z

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

    Z_2_below = np.zeros((2*dim, 2*dim))
    def block_2_below_zeros(X):
        return batch_equals(X, Z_2_below, 4*dim, 6*dim, 2*dim, 4*dim)
    def grad_block_2_below_zeros(X):
        return batch_equals_grad(X, Z_2_below, 4*dim, 6*dim, 2*dim, 4*dim)
    Z_2_right = np.zeros((2*dim, 2*dim))
    def block_2_right_zeros(X):
        return batch_equals(X, Z_2_right, 2*dim, 4*dim, 4*dim, 6*dim)
    def grad_block_2_right_zeros(X):
        return batch_equals_grad(X, Z_2_right, 2*dim, 4*dim, 4*dim, 6*dim)

    Z_3_below = np.zeros((dim, dim))
    def block_3_below_zeros(X):
        return batch_equals(X, Z_3_below, 5*dim, 6*dim, 4*dim, 5*dim)
    def grad_block_3_below_zeros(X):
        return batch_equals(X, Z_3_below, 5*dim, 6*dim, 4*dim, 5*dim)
    Z_3_right = np.zeros((dim, dim))
    def block_3_right_zeros(X):
        return batch_equals(X, Z_3_right, 4*dim, 5*dim, 5*dim, 6*dim)
    def grad_block_3_below_zeros(X):
        return batch_equals(X, Z_3_right, 4*dim, 5*dim, 5*dim, 6*dim)


    Gs += [block_1_below_zeros, block_1_right_zeros,
            block_2_below_zeros, block_2_right_zeros,
            block_3_below_zeros, block_3_right_zeros]
    gradGs += [grad_block_1_below_zeros, grad_block_1_right_zeros,
            grad_block_2_below_zeros, grad_block_2_right_zeros,
            grad_block_3_below_zeros, grad_block_3_right_zeros]


if __name__ == "__main__":
    #test1()
    testQ()
    pass

