from hazan import *

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
    Solves a simple version of the SDPs required for
    Q optimization

    Given
     A,
     F = B^{.5},

    We want to solve for x = [s vec(Z) vec(Q)]. To do so, we
    construct and solve the following optimization problem

    max Tr Z + log det Q

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
    4 * dim by 4 * dim

    TODO: Think of how to shrink this representation
        Ideas:
            1) Add specific zero penalty operations so
               we don't need to do python for-loops. ====> Done
    """
    dim = 2
    cdim = 4 * dim
    N_iter = 50
    g = GeneralSDPHazanSolver()
    As = []
    bs = []
    Cs = []
    ds = []
    Fs = []
    gradFs = []
    Gs = []
    gradGs = []
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
    Q_coords = (4*dim, 5*dim, 4*dim, 5*dim)
    block_1_Q_coords = (dim, 2*dim, dim, 2*dim)
    Z = zeros((dim, dim))
    def block_1_Q(X):
        return batch_linear_equals(X, c, Q_coords, Z, block_1_Q_coords)
    def grad_block_1_Q(X):
        return grad_batch_linear_equals(X, c, Q_coords, Z,
                grad_block_1_Q_coords)
    d = -1.
    block_2_Q_coords = (2*dim, 3*dim, 2*dim, 3*dim)
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
            else:
                # Swap this out for a sparse representation ...
                Cij = np.zeros((dim, dim))
                Cij[i,j] = 1.
                dij = 0.
                Cs.append(Cij)
                ds.append(dij)
                pass


    ds = []

if __name__ == "__main__":
    test1()
    pass

