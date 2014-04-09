from hazan import *

# Do a simple test of General SDP Solver with binary search

def test1():
    # Check argument validation
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
    # Now try a simple problem
    g = GeneralSDPHazanSolver()
    As = [np.array([[1., 2.],
                    [1., 2.]])]
    bs = [1]
    Cs = []
    ds = []
    E = np.array([[1., 0.],
                  [0., 1.]])
    R = 10.
    upper, lower, X_upper, X_lower, fail = g.solve(E, As, bs, Cs, ds,
                                                eps, dim, R)
