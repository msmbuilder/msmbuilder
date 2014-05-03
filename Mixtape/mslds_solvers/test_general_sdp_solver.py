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
    N_iter = 50
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
    R = 1.
    eps = 1./N_iter
    Fs = []
    gradFs = []
    Gs = []
    gradGs = []
    U = 2
    L = 0
    upper, lower, X_upper, X_lower, SUCCEED = g.solve(h, gradh, As, bs,
                Cs, ds, Fs, gradFs, Gs, gradGs, eps, dim, R, U, L)
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

    max s*dim + Tr Z

          ---------------------------
         |Z+sI   F                   |
         |F.T    Q                   |
         |           D-Q   A         |
    X =  |           A.T D^{-1}      |
         |                      Q    |
         |                        Z  |
         |                          s|
          ---------------------------
    X is PSD

    If Q is dim by dim, then this matrix is
    (4 * dim + 1) by (4 * dim + 1)

    TODO: Think of how to shrink this representation
        Ideas:
            1) Add specific zero penalty operations so
               we don't need to do python for-loops.
    """
    qdim = 2
    dim = 4 * qdim
    N_iter = 50
    g = GeneralSDPHazanSolver()
    As = []
    bs = []
    Cs = []
    for i in range(dim):
        for j in range(dim):
            if i < qdim and j < qdim:
            #  ---------
            # |Z+sI   F |
            # |F.T    Q |
            #  ---------
                pass
            elif ((qdim <= i) and (i < 2 * qdim) and
                  (qdim <= j) and (j < 2 * qdim)):
            #  -----------
            # | D-Q   A   |
            # | A.T D^{-1}|
            #  -----------
                pass
            elif ((2 * qdim <= i) and (i < 3 * qdim) and
                  (2 * qdim <= j) and (j < 3 * qdim)):
            # ---
            #| Q |
            # ---
                pass
            elif ((3 * qdim <= i) and (i < 4 * qdim) and
                (3 * qdim <= j) and (j < 4 * qdim)):
                pass
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

