import sys
sys.path.append("..")
from hazan import *
from test_bounded_trace_sdp_solver import batch_equality
import pdb

def test1():
    """
    Check that the feasibility solver with one inequality reports that a
    feasible problem is feasible.  With As, bs defined as below, the
    constraints translate to

    x_11 + 2 x_22 <= 1.5
    Tr(X) = x_11 + x_22 == 1.

    These two equations are simultaneously satisfiable.
    """
    dim = 2
    f = FeasibilitySDPHazanSolver()
    As = [np.array([[1, 0.],
                    [0., 2]])]
    bs = [1.5]
    eps = 1e-1
    Cs = []
    ds = []
    Fs = []
    gradFs = []
    Gs = []
    gradGs = []
    X, fX, SUCCEED = f.feasibility_solve(As, bs, Cs, ds, Fs, gradFs,
            Gs, gradGs, eps, dim)
    assert SUCCEED == True

def test2():
    """
    Check that the feasibility solver with one inequality reports that
    an infeasible problem is infeasible.  With As, bs defined as below,
    the constraints translate to

     2 x_11 + 2 x_22 <= 1.5
     Tr(X) = x_11 + x_22 == 1.

    These two equations are not simultaneously satisfiable.
    """
    # Now try two-dimensional basic infeasibility example
    f = FeasibilitySDPHazanSolver()
    As = [np.array([[2, 0.],
                    [0., 2]])]
    bs = [1.]
    eps = 1e-1
    dim = 2
    Cs = []
    ds = []
    Fs = []
    gradFs = []
    Gs = []
    gradGs = []
    X, fX, SUCCEED = f.feasibility_solve(As, bs, Cs, ds, Fs, gradFs,
            Gs, gradGs, eps, dim)
    assert SUCCEED == False

def test3():
    """
    Check that the feasibility solver with one equality constraint reports
    that a feasible problem is feasible.  With C, d defined as below, the
    constraints translate to

    x_11 + 2 x_22 = 1.5
    Tr(X) = x_11 + x_22 == 1.

    These two equations are simultaneously satisfiable.
    """
    fudge_factor = 5.0
    N_iter = 50
    # Now try two-dimensional basic feasible example
    f = FeasibilitySDPHazanSolver()
    As = []
    bs = []
    Cs = [np.array([[1, 0.],
                    [0., 2]])]
    ds = [1.5]
    eps = 1./N_iter
    dim = 2
    Fs = []
    gradFs = []
    Gs = []
    gradGs = []
    X, fX, SUCCEED = f.feasibility_solve(As, bs, Cs, ds, Fs, gradFs,
            Gs, gradGs, eps, dim)
    assert SUCCEED == True
    assert np.abs(X[0,0] + 2 * X[1,1] -1.5) < fudge_factor * eps
    assert np.abs(X[0,0] + X[1,1] - 1) < fudge_factor * eps

def test4():
    """
    Check that the feasibility solver discerns feasibility of batch
    equality problems like

    feasibility(X)
    subject to
        [[ B   , A],
         [ A.T , D]]  is PSD, where B, D are arbitrary, A given.
        Tr(X) = Tr(B) + Tr(D) == 1
    """
    dims = [8]
    N_iter = 400
    f = FeasibilitySDPHazanSolver()
    for dim in dims:
        A = (1./(2*dim)) * np.eye(int(dim/2))
        print "A\n", A
        M, As, bs, Cs, ds, Fs, gradFs, Gs, gradGs, eps = \
                batch_equality(A, dim, N_iter)
        X, fX, SUCCEED = f.feasibility_solve(As, bs, Cs, ds, Fs, gradFs,
                Gs, gradGs, eps, dim)
        #import pdb
        #pdb.set_trace()
        #assert SUCCEED == True

if __name__ == "__main__":
    #test1()
    #test2()
    #test3()
    #test4()
    pass
