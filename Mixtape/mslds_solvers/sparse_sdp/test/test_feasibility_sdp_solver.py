import sys
sys.path.append("..")
from feasibility_sdp_solver import FeasibilitySolver
from constraints import *

def test1():
    """
    Test
    feasibility(X):
        x_11 + 2 x_22 == 1.5
        Tr(X) <= R

    These two equations are simultaneously satisfiable for R >= 0.75
    """
    eps = 1e-3
    tol = 1e-2
    N_iter = 50
    Rs = [1, 10, 100, 1000]
    for R in Rs:
        dim, As, bs, Cs, ds, Fs, gradFs, Gs, gradGs = \
               simple_equality_constraint()
        f = FeasibilitySolver(R, dim, eps)
        f.init_solver(As, bs, Cs, ds, Fs, gradFs, Gs, gradGs)
        X, fX, succeed = f.feasibility_solve(N_iter, tol,
                methods=['frank_wolfe'])
        assert succeed == True

def test2():
    """
    Test the following problem is infeasible

     x_11 + 2 x_22 == 1.5
     Tr(X) = x_11 + x_22 <= R

    These two equations are not simultaneously satisfiable for small R.
    """
    # Now try two-dimensional basic infeasibility example
    eps = 1e-3
    tol = 1e-2
    N_iter = 50
    Rs = [0.1, 0.25, 0.5]
    for R in Rs:
        dim, As, bs, Cs, ds, Fs, gradFs, Gs, gradGs = \
               simple_equality_constraint()
        f = FeasibilitySolver(R, dim, eps)
        f.init_solver(As, bs, Cs, ds, Fs, gradFs, Gs, gradGs)
        X, fX, succeed = f.feasibility_solve(N_iter, tol,
                methods=['frank_wolfe'])
        assert succeed == False

def test3():
    """
    Check feasibility with simple equality and inequality constraints.

        feasbility(X)
        subject to
            x_11 + 2 x_22 <= 1
            x_11 + 2 x_22 + 2 x_33 == 5/3
            Tr(X) <= R

    These two equations are simultaneously satisfiable.
    """
    eps = 1e-3
    tol = 1e-2
    N_iter = 50
    Rs = [1]
    for R in Rs:
        dim, As, bs, Cs, ds, Fs, gradFs, Gs, gradGs = \
               simple_equality_constraint()
        f = FeasibilitySolver(R, dim, eps)
        f.init_solver(As, bs, Cs, ds, Fs, gradFs, Gs, gradGs)
        X, fX, succeed = f.feasibility_solve(N_iter, tol,
                methods=['frank_wolfe'])
        assert succeed == True

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
