from hazan import *

def test1():
    # Do a simple test of the feasibility solver
    dim = 2

    # Check argument validation
    Error = False
    try:
        g = GeneralSDPHazanSolver()
        As = [np.array([[1.5, 0.],
                        [0., 1.5]])]
        bs = [np.array([1.5, 0])]
        Cs = []
        ds = []
        eps = 1e-1
        dim = 1
        g.feasibility_solve(As, bs, Cs, ds, eps, dim)
    except ValueError:
        Error = True
    assert Error == True

def test2():
    # Now try two-dimensional basic feasible example
    g = GeneralSDPHazanSolver()
    As = [np.array([[1, 0.],
                    [0., 2]])]
    bs = [1.5]
    eps = 1e-1
    dim = 2
    Cs = []
    ds = []
    X, fX, FAIL = g.feasibility_solve(As, bs, Cs, ds, eps, dim)
    assert FAIL == False

def test3():
    # Now try two-dimensional basic infeasibility example
    g = GeneralSDPHazanSolver()
    As = [np.array([[2, 0.],
                    [0., 2]])]
    bs = [1.]
    eps = 1e-1
    dim = 2
    Cs = []
    ds = []
    X, fX, FAIL = g.feasibility_solve(As, bs, Cs, ds, eps, dim)
    assert FAIL == True

if __name__ == "__main__":
    test1()
    test2()
    test3()
