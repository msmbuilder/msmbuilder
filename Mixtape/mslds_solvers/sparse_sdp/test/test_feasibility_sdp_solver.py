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
    Rs = [1, 10, 100]
    for R in Rs:
        dim, As, bs, Cs, ds, Fs, gradFs, Gs, gradGs = \
               simple_equality_and_inequality_constraint()
        f = FeasibilitySolver(R, dim, eps)
        f.init_solver(As, bs, Cs, ds, Fs, gradFs, Gs, gradGs)
        X, fX, succeed = f.feasibility_solve(N_iter, tol,
                methods=['frank_wolfe'])
        assert succeed == True

def test4():
    """
    test4: feasibility of batch equality

    feasibility(X)
    subject to
        [[ B   , A],
         [ A.T , D]]  is PSD, where B, D are arbitrary, A given.
        Tr(X) = Tr(B) + Tr(D) == 1
    """
    eps = 1e-5
    tol = 1e-2
    Rs = [10]
    dims = [8]
    N_iter = 200
    for R in Rs:
        for dim in dims:
            block_dim = int(dim/2)
            A = (1./dim)*np.eye(block_dim)
            B = np.eye(block_dim)
            D = np.eye(block_dim)
            tr_B_D = np.trace(B) + np.trace(D)
            B = B / tr_B_D
            D = D / tr_B_D
            As, bs, Cs, ds, Fs, gradFs, Gs, gradGs = \
                    basic_batch_equality(dim, A, B, D)
            f = FeasibilitySolver(R, dim, eps)
            f.init_solver(As, bs, Cs, ds, Fs, gradFs, Gs, gradGs)
            #import pdb
            #pdb.set_trace()
            X, fX, succeed = f.feasibility_solve(N_iter, tol,
                    methods=['frank_wolfe', 'frank_wolfe_stable'])
                    #    'projected_gradient'])
            assert succeed == True
