from hazan import *
import pdb

def test1():
    """
    Check that argument validation is working.
    """
    dim = 2

    # Check argument validation
    Error = False
    try:
        g = GeneralSDPHazanSolver()
        As = [np.array([[1.5, 0.],
                        [0., 1.5]])]
        # bs is misformatted; should be bs = [1.5] in this case
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
    """
    Check that the feasibility solver with just inequalities reports that a
    feasible problem is feasible.
    """
    # Now try two-dimensional basic feasible example
    g = GeneralSDPHazanSolver()
    # With A defined as below, the constraints translate to
    # x_11 + 2 x_22 <= 1.5
    # the unit trace constraint is
    # x_11 + x_22 = 1.
    # these two equations can both be true
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
    """
    Check that the feasibility solver with just inequalities reports that
    an infeasible problem is infeasible.
    """
    # Now try two-dimensional basic infeasibility example
    g = GeneralSDPHazanSolver()
    # With A defined as below, the constraints translate to
    # 2 x_11 + 2 x_22 <= 1.5
    # the unit trace constraint is
    # x_11 + x_22 = 1.
    # these two equations cannot both be true
    As = [np.array([[2, 0.],
                    [0., 2]])]
    bs = [1.]
    eps = 1e-1
    dim = 2
    Cs = []
    ds = []
    X, fX, FAIL = g.feasibility_solve(As, bs, Cs, ds, eps, dim)
    assert FAIL == True

def test4():
    """
    Check that the feasibility solver with just equalities reports that
    a feasible problem is feasible.
    """
    # Now try two-dimensional basic feasible example
    g = GeneralSDPHazanSolver()
    # With C, d defined as below, the constraints translate to
    # x_11 + 2 x_22 = 1.5
    # the unit trace constraint is
    # x_11 + x_22 = 1.
    # these two equations specify a unique solution
    As = []
    bs = []
    Cs = [np.array([[1, 0.],
                    [0., 2]])]
    ds = [1.5]
    eps = 1e-1
    dim = 2
    X, fX, FAIL = g.feasibility_solve(As, bs, Cs, ds, eps, dim)
    assert FAIL == False
    assert np.abs(X[0,0] + 2 * X[1,1] -1.5) < eps
    assert np.abs(X[0,0] + X[1,1] - 1) < eps

if __name__ == "__main__":
    #test1()
    #test2()
    #test3()
    test4()
