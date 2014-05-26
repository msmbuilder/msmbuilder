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
    Rs = [10, 100, 1000]
    dim, As, bs, Cs, ds, Fs, gradFs, Gs, gradGs = \
           simple_equality_constraint()
    f = FeasibilitySolver(dim, eps, As, bs, Cs, ds, Fs, gradFs, Gs, gradGs)
    X, fX, succeed = f.feasibility_solve(N_iter, tol,
            methods=['frank_wolfe'], disp=False, Rs=Rs)
    assert succeed == True

def test1b():
    """
    Test
    feasibility(X):
        x_11 + 2 x_22 == 50
        Tr(X) <= R

    These two equations are simultaneously satisfiable for R >= 0.75
    """
    eps = 1e-3
    tol = 1e-2
    N_iter = 50
    Rs = [10, 100, 1000]
    dim, As, bs, Cs, ds, Fs, gradFs, Gs, gradGs = \
           simple_equality_constraint()
    ds = [50.]
    f = FeasibilitySolver(dim, eps, As, bs, Cs, ds, Fs, gradFs, Gs, gradGs)
    X, fX, succeed = f.feasibility_solve(N_iter, tol,
            methods=['frank_wolfe'], disp=True, Rs=Rs)
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
    dim, As, bs, Cs, ds, Fs, gradFs, Gs, gradGs = \
           simple_equality_constraint()
    f = FeasibilitySolver(dim, eps, As, bs, Cs, ds, Fs, gradFs, Gs, gradGs)
    X, fX, succeed = f.feasibility_solve(N_iter, tol,
            methods=['frank_wolfe'], disp=True, Rs=Rs)
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
    dim, As, bs, Cs, ds, Fs, gradFs, Gs, gradGs = \
           simple_equality_and_inequality_constraint()
    f = FeasibilitySolver(dim, eps, As, bs, Cs, ds, Fs, gradFs, Gs, gradGs)
    X, fX, succeed = f.feasibility_solve(N_iter, tol,
            methods=['frank_wolfe'], disp=True, Rs=Rs)
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
        f = FeasibilitySolver(dim, eps, As, bs, Cs, ds,
                Fs, gradFs, Gs, gradGs)
        X, fX, succeed = f.feasibility_solve(N_iter, tol,
                methods=['frank_wolfe', 'frank_wolfe_stable'],
                disp=True, Rs=Rs)
        assert succeed == True

def test5():
    """
    Tests feasibility Q optimization.

    feasibility(X)
          --------------
         |D-ADA.T  I    |
    X =  |   I     R    |
         |            R |
          --------------
    X is PSD
    """
    dims = [4]
    eps = 1e-5
    tol = 1e-2
    Rs = [10]
    N_iter = 100
    for dim in dims:
        block_dim = int(dim/4)

        # Generate initial data
        D = np.eye(block_dim)
        Dinv = np.linalg.inv(D)
        B = np.eye(block_dim)
        A = 0.5*(1./dim) * np.eye(block_dim)
        gamma = .5
        c = np.sqrt(1/gamma)
        As, bs, Cs, ds, Fs, gradFs, Gs, gradGs = \
                Q_constraints(block_dim, A, B, D, c)
        (D_ADA_T_cds, I_1_cds, I_2_cds, R_1_cds, 
            D_cds, c_I_1_cds, c_I_2_cds, R_2_cds) = \
                Q_coords(block_dim)
        f = FeasibilitySolver(dim, eps, As, bs, Cs, ds,
                Fs, gradFs, Gs, gradGs)
        X, fX, succeed = f.feasibility_solve(N_iter, tol,
                methods=['frank_wolfe', 'frank_wolfe_stable'],
                disp=False, Rs=Rs)
        assert succeed == True

def test6():
    """
    Tests feasibility of A optimization.

    feasibility(X)
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
    dims = [8]
    eps = 1e-5
    tol = 1e-2
    Rs = [10]
    N_iter = 100
    for dim in dims:
        block_dim = int(dim/4)

        # Generate random data
        D = np.eye(block_dim)
        Dinv = np.linalg.inv(D)
        Q = 0.5 * np.eye(block_dim)
        C = 2 * np.eye(block_dim)
        B = np.eye(block_dim)
        E = np.eye(block_dim)
        mu = np.random.rand(block_dim)

        As, bs, Cs, ds, Fs, gradFs, Gs, gradGs = \
                A_constraints(block_dim, D, Dinv, Q, mu)

        (D_Q_cds, Dinv_cds, I_1_cds, I_2_cds,
            A_1_cds, A_T_1_cds, A_2_cds, A_T_2_cds) = A_coords(block_dim)
        f = FeasibilitySolver(dim, eps, As, bs, Cs, ds,
                Fs, gradFs, Gs, gradGs)
        X, fX, succeed = f.feasibility_solve(N_iter, tol,
                methods=['frank_wolfe', 'frank_wolfe_stable'],
                disp=False, Rs=Rs)
        assert succeed == True
