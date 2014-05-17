import sys
sys.path.append("..")
from general_sdp_solver import *
from objectives import *
from constraints import *
import scipy
import numpy as np

# Do a simple test of General SDP Solver with binary search

def test1():
    """
    A simple semidefinite program

    min Tr(X)
    subject to
        x_11 + 2 x_22 == 1
        Tr(X) = x_11 + x_22 <= 10
        X semidefinite

    The solution to this problem is

        X = [[0, 0],
             [0, .75]]

    from Lagrange multiplier.
    """
    eps = 1e-4
    tol = 1e-3
    search_tol = 1e-2
    N_iter = 50
    dim = 2
    Rs = [10]
    L, U = (-10, 10)
    for R in Rs:
        dim, As, bs, Cs, ds, Fs, gradFs, Gs, gradGs = \
                simple_equality_constraint()
        g = GeneralSolver(R, L, U, dim, eps)
        g.save_constraints(trace_obj, grad_trace_obj, As, bs, Cs, ds,
                Fs, gradFs, Gs, gradGs)
        (alpha, _, _, _, _, succeed) = g.solve(N_iter, tol,
                interactive=False, disp=True)
        assert succeed == True
        assert np.abs(alpha - 0.75) < search_tol

def test2():
    """
    Tests feasibility Q optimization.

    min_Q -log det R + Tr(RB)
          --------------
         |D-ADA.T  I    |
    X =  |   I     R    |
         |            R |
          --------------
    X is PSD
    """
    eps = 1e-4
    tol = 1e-3
    search_tol = 1e-2
    N_iter = 50
    dim = 2
    Rs = [10]
    dims = [3]
    L, U = (-10, 10)
    for R in Rs:
        for dim in dims:
            block_dim = int(dim/3)

            # Generate initial data
            D = np.eye(block_dim)
            Dinv = np.linalg.inv(D)
            B = np.eye(block_dim)
            A = 0.5*(1./dim) * np.eye(block_dim)
            As, bs, Cs, ds, Fs, gradFs, Gs, gradGs = \
                    Q_constraints(block_dim, A, B, D)
            g = GeneralSolver(R, L, U, dim, eps)
            def obj(X):
                return log_det_tr(X, B)
            def grad_obj(X):
                return grad_log_det_tr(X, B)
            g.save_constraints(obj, grad_obj, As, bs, Cs, ds,
                    Fs, gradFs, Gs, gradGs)
            (alpha, _, _, _, _, succeed) = g.solve(N_iter, tol,
                    disp=True, interactive=False)
            assert succeed == True

def test3():
    """
    Tests feasibility of A optimization.

    min_A Tr [ Q^{-1} ([C - B] A.T + A [C - B].T + A E A.T]

          --------------------
         | D-Q    A           |
    X =  | A.T  D^{-1}        |
         |              I   A |
         |             A.T  I |
          --------------------
    X is PSD

    If A is dim by dim, then this matrix is 4 * dim by 4 * dim.
    The solution to this problem is A = 0 when dim = 1.
    """
    eps = 1e-4
    tol = 1e-3
    search_tol = 1e-2
    N_iter = 100
    Rs = [5]
    dims = [4]
    L, U = (-10, 10)
    for R in Rs:
        for dim in dims:
            block_dim = int(dim/4)

            # Generate random data
            D = np.eye(block_dim)
            Dinv = np.linalg.inv(D)
            Q = 0.5 * np.eye(block_dim)
            Qinv = np.linalg.inv(Q)
            C = 2 * np.eye(block_dim)
            B = np.eye(block_dim)
            E = np.eye(block_dim)

            As, bs, Cs, ds, Fs, gradFs, Gs, gradGs = \
                    A_constraints(block_dim, D, Dinv, Q)

            def obj(X):
                return A_dynamics(X, block_dim, C, B, E, Qinv)
            def grad_obj(X):
                return grad_A_dynamics(X, block_dim, C, B, E, Qinv)
            g = GeneralSolver(R, L, U, dim, eps)
            g.save_constraints(obj, grad_obj, As, bs, Cs, ds,
                    Fs, gradFs, Gs, gradGs)
            (alpha, _, _, _, _, succeed) = g.solve(N_iter, tol,
                    disp=True, interactive=False)
            assert succeed == True

def test4():
    """
    Tests feasibility Q optimization with realistic values for F,
    D, A from the 1-d 2-well toy system.


    min_Q -log det R + Tr(RF)
          --------------
         |D-ADA.T  I    |
    X =  |   I     R    |
         |            R |
          --------------
    X is PSD
    """
    eps = 1e-4
    tol = 5e-2
    search_tol = 1e-2
    N_iter = 150
    Rs = [10]
    dims = [3]
    L, U = (0, 1000)
    scale = 10.
    for R in Rs:
        for dim in dims:
            block_dim = int(dim/3)

            # Generate initial data
            D = .0204 * np.eye(block_dim)
            F = 25.47 * np.eye(block_dim)
            A = np.zeros(block_dim)
            # Rescaling
            D *= scale
            As, bs, Cs, ds, Fs, gradFs, Gs, gradGs = \
                    Q_constraints(block_dim, A, F, D)
            g = GeneralSolver(R, L, U, dim, eps)
            def obj(X):
                return log_det_tr(scale*X, F)
            def grad_obj(X):
                return grad_log_det_tr(scale*X, F)
            g.save_constraints(obj, grad_obj, As, bs, Cs, ds,
                    Fs, gradFs, Gs, gradGs)
            (alpha, U, X_U, L, X_L, succeed) = g.solve(N_iter, tol,
                    verbose=True, interactive=True)
            (D_ADA_T_cds, I_1_cds, I_2_cds, R_1_cds, R_2_cds) \
                    = Q_coords(block_dim)
            # Undo trace scaling
            X_L, X_U = R*X_L, R*X_U
            print "X_L\n", X_L
            print "X_U\n", X_U
            if X_L != None:
                R_1_L  = get_entries(X_L, R_1_cds)
                R_2_L  = get_entries(X_L, R_2_cds)
                # Undo scaling
                R_1_L, R_2_L = scale*R_1_L, scale*R_2_L
                Q_1_L = np.linalg.inv(R_1_L)
                print "Q_1_L:\n", Q_1_L
                Q_2_L = np.linalg.inv(R_2_L)
                print "Q_2_L:\n", Q_2_L
            if X_U != None:
                R_1_U  = get_entries(X_U, R_1_cds)
                R_2_U  = get_entries(X_U, R_2_cds)
                # Undo scaling
                R_1_U, R_2_U = scale*R_1_U, scale*R_2_U
                Q_1_U = np.linalg.inv(R_1_U)
                print "Q_1_U:\n", Q_1_U
                Q_2_U = np.linalg.inv(R_2_U)
                print "Q_2_U:\n", Q_2_U
            if X_L != None:
                assert np.linalg.norm(Q_1_L, 2) < np.linalg.norm(D)
                assert np.linalg.norm(Q_2_L, 2) < np.linalg.norm(D)
            if X_U != None:
                assert np.linalg.norm(Q_1_U, 2) < np.linalg.norm(D)
                assert np.linalg.norm(Q_2_U, 2) < np.linalg.norm(D)
            assert succeed == True
