from hazan import *

# Do a simple test of General SDP Solver with binary search

def test1():
    """
    Check argument validation
    """
    Error = False
    try:
        g = GeneralSDPHazanSolver()
        As = [np.array([[1., 2.],
                        [1., 2.]])]
        bs = [np.array([1., 1.])]
        Cs = []
        ds = []
        E = np.array([[1.],
                      [0.]])
        eps = 1e-1
        dim = 2
        R = 10
        g.solve(E, As, bs, Cs, ds, eps, dim, R)
    except ValueError:
        Error = True
    assert Error == True

def test2():
    """
    A simple semidefinite program

    max x_11 + x_22
    subject to
        x_11 + 2 x_22 == 1
        x_12 == x_21 == 0
        X semidefinite

    The solution to this problem is

        X = [[1, 0],
             [0, 0]]

    from Lagrange multipliers (or just directly).
    """
    dim = 2
    N_iter = 50
    g = GeneralSDPHazanSolver()
    As = []
    bs = []
    Cs = [np.array([[1., 0.],
                    [0., 2.]]),
          np.array([[0., 1.],
                    [0., 0.]]),
          np.array([[0., 0.],
                    [1., 0.]])]
    ds = [1., 0., 0.]
    E = np.array([[1., 0.],
                  [0., 1.]])
    R = 1.
    eps = 1./N_iter
    upper, lower, X_upper, X_lower, SUCCEED = g.solve(E, As, bs, Cs, ds,
                                                eps, dim, R)
    print
    print "General SDP Solver Finished"
    print "SUCCEED: ", SUCCEED
    print "upper: ", upper
    print "X_upper:\n", X_upper
    print "X_lower:\n", X_lower
    print "lower: ", lower

if __name__ == "__main__":
    #test1()
    test2()

