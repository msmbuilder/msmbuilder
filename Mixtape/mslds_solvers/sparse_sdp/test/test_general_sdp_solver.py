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
    Rs = [10, 100, 1000]
    dim, As, bs, Cs, ds, Fs, gradFs, Gs, gradGs = \
            simple_equality_constraint()
    g = GeneralSolver(dim, eps)
    g.save_constraints(trace_obj, grad_trace_obj, As, bs, Cs, ds,
            Fs, gradFs, Gs, gradGs)
    (L, U, X, succeed) = g.solve(N_iter, tol, verbose=False,
            interactive=False, disp=True, debug=True, Rs = Rs)
    print "X:\n", X
    assert succeed == True
    assert np.abs(X[1,1] - 0.75) < search_tol

def test2():
    """
    A simple semidefinite program to test trace search
    min Tr(X)
    subject to
        x_11 + 2 x_22 == 50
        X semidefinite

    The solution to this problem is

        X = [[0, 0],
             [0, 25]]
    """
    eps = 1e-4
    tol = 1e-2
    search_tol = 1e-2
    N_iter = 50
    Rs = [10, 100]
    dim = 2
    As, bs = [], []
    Cs = [np.array([[ 1.,  0.],
                    [ 0.,  2.]])]
    ds = [50]
    Fs, gradFs, Gs, gradGs = [], [], [], []
    g = GeneralSolver(dim, eps)
    g.save_constraints(trace_obj, grad_trace_obj, As, bs, Cs, ds,
            Fs, gradFs, Gs, gradGs)
    (L, U, X, succeed) = g.solve(N_iter, tol, verbose=False,
            interactive=False, debug=False, Rs = Rs)
    print "X:\n", X
    assert succeed == True
    assert np.abs(np.trace(X) - 25) < search_tol

def test3():
    """
    A simple quadratic program
    min x_1
    subject to
        x_1^2 + x_2^2 = 1

    The solution to this problem is

        X = [[ 0, 0],
             [ 0, 1]]
        X semidefinite
    """
    eps = 1e-4
    tol = 1e-2
    search_tol = 3e-2 # Figure out how to reduce this...
    N_iter = 50
    Rs = [10, 100]
    dim = 2
    As, bs, Cs, ds, Fs, gradFs = [], [], [], [], [], []
    g = lambda(X): X[0,0]**2 + X[1,1]**2 - 1.
    def gradg(X):
        (dim, _) = np.shape(X)
        grad = np.zeros(np.shape(X))
        grad[range(dim), range(dim)] = 2*X[range(dim), range(dim)]
        return grad
    Gs, gradGs = [g], [gradg]
    def obj(X):
        return X[0,0]
    def grad_obj(X):
        G = np.zeros(np.shape(X))
        G[0,0] = 1.
        return G
    g = GeneralSolver(dim, eps)
    g.save_constraints(obj, grad_obj, As, bs, Cs, ds,
            Fs, gradFs, Gs, gradGs)
    (L, U, X, succeed) = g.solve(N_iter, tol, verbose=False,
            interactive=False, debug=False, Rs = Rs)
    print "X:\n", X
    assert succeed == True
    assert np.abs(X[0,0] - 0) < search_tol

def test4():
    """
    Tests that feasibility of Q optimization runs.

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
    Rs = [10, 100]
    dims = [3]
    for dim in dims:
        block_dim = int(dim/3)

        # Generate initial data
        D = np.eye(block_dim)
        Dinv = np.linalg.inv(D)
        B = np.eye(block_dim)
        A = 0.5*(1./dim) * np.eye(block_dim)
        As, bs, Cs, ds, Fs, gradFs, Gs, gradGs = \
                Q_constraints(block_dim, A, B, D)
        g = GeneralSolver(dim, eps)
        def obj(X):
            return log_det_tr(X, B)
        def grad_obj(X):
            return grad_log_det_tr(X, B)
        g.save_constraints(obj, grad_obj, As, bs, Cs, ds,
                Fs, gradFs, Gs, gradGs)
        (L, U, X, succeed) = g.solve(N_iter, tol,
                disp=True, interactive=False, debug=True, Rs=Rs)
        assert succeed == True

def test5():
    """
    Tests feasibility of A optimization.

    min_A Tr [ Q^{-1} ([C - B] A.T + A [C - B].T + A E A.T]

          --------------------
         | D-Q    A           |
    X =  | A.T  D^{-1}        |
         |              I   A |
         |             A.T  I |
          --------------------
    A mu == 0
    X is PSD

    If A is dim by dim, then this matrix is 4 * dim by 4 * dim.
    The solution to this problem is A = 0 when dim = 1.
    """
    eps = 1e-4
    tol = 1e-3
    search_tol = 1e-2
    N_iter = 100
    Rs = [10, 100]
    dims = [4]
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
        mu = np.ones((block_dim, 1))

        As, bs, Cs, ds, Fs, gradFs, Gs, gradGs = \
                A_constraints(block_dim, D, Dinv, Q, mu)
        def obj(X):
            return A_dynamics(X, block_dim, C, B, E, Qinv)
        def grad_obj(X):
            return grad_A_dynamics(X, block_dim, C, B, E, Qinv)
        g = GeneralSolver(dim, eps)
        g.save_constraints(obj, grad_obj, As, bs, Cs, ds,
                Fs, gradFs, Gs, gradGs)
        (L, U, X, succeed) = g.solve(N_iter, tol,
                disp=True, interactive=False, Rs=Rs)
        assert succeed == True

def test6():
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
    tol = 1e-2
    search_tol = 1e-2
    N_iter = 150
    Rs = [10, 100]
    dims = [3]
    scale = 10.
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
        g = GeneralSolver(dim, eps)
        def obj(X):
            return log_det_tr(scale*X, F)
        def grad_obj(X):
            return grad_log_det_tr(scale*X, F)
        g.save_constraints(obj, grad_obj, As, bs, Cs, ds,
                Fs, gradFs, Gs, gradGs)
        (L, U, X, succeed) = g.solve(N_iter, tol, verbose=False,
                interactive=False, debug=False, Rs=Rs)
        (D_ADA_T_cds, I_1_cds, I_2_cds, R_1_cds, R_2_cds) \
                = Q_coords(block_dim)
        # Undo trace scaling
        if X != None:
            print "X\n", X
            R_1  = scale*get_entries(X, R_1_cds)
            R_2  = scale*get_entries(X, R_2_cds)
            Q_1 = np.linalg.inv(R_1)
            Q_2 = np.linalg.inv(R_2)
            print "Q_1:\n", Q_1
            print "Q_2:\n", Q_2
            # Undo rescaling
            D *= (1./scale)
            if X != None:
                print "nplinalg.norm(D, 2): ", np.linalg.norm(D, 2)
                assert np.linalg.norm(Q_1, 2) < 1.1*np.linalg.norm(D, 2)
                assert np.linalg.norm(Q_2, 2) < 1.1*np.linalg.norm(D, 2)
        assert succeed == True

def test7():
    """
    Tests feasibility of A optimization.

    min_A Tr [ Q^{-1} ([C - B] A.T + A [C - B].T + A E A.T]

          --------------------
         | D-Q    A           |
    X =  | A.T  D^{-1}        |
         |              I   A |
         |             A.T  I |
          --------------------
    A mu == 0
    X is PSD

    If A is dim by dim, then this matrix is 4 * dim by 4 * dim.
    The solution to this problem is A = 0 when dim = 1.
    """
    eps = 1e-4
    tol = 1e-2
    search_tol = 1e-2
    N_iter = 100
    Rs = [10, 100]
    dims = [4]
    scale = 10.
    for dim in dims:
        block_dim = int(dim/4)

        # Generate random data
        D = .0204 * np.eye(block_dim)
        Dinv = np.linalg.inv(D)
        Q = .02 * np.eye(block_dim)
        Qinv = np.linalg.inv(Q)
        C = 1225.025 * np.eye(block_dim)
        B = 1238.916 * np.eye(block_dim)
        E = 48.99 * np.eye(block_dim)
        mu = np.ones((block_dim, 1))

        # Rescaling
        D *= scale
        Q *= scale
        Dinv *= (1./scale)
        Qinv *= (1./scale)
        As, bs, Cs, ds, Fs, gradFs, Gs, gradGs = \
                A_constraints(block_dim, D, Dinv, Q, mu)
        (D_Q_cds, Dinv_cds, I_1_cds, I_2_cds,
            A_1_cds, A_T_1_cds, A_2_cds, A_T_2_cds) = A_coords(block_dim)

        def obj(X):
            return A_dynamics(X, block_dim, C, B, E, Qinv)
        def grad_obj(X):
            return grad_A_dynamics(X, block_dim, C, B, E, Qinv)
        g = GeneralSolver(dim, eps)
        g.save_constraints(obj, grad_obj, As, bs, Cs, ds,
                Fs, gradFs, Gs, gradGs)
        (L, U, X, succeed) = g.solve(N_iter, tol,
                verbose=False, disp=True, interactive=False, Lmin=-100)
        # Undo trace scaling
        print "X\n", X
        if X != None:
            A_1 = get_entries(X, A_1_cds)
            A_T_1 = get_entries(X, A_T_1_cds)
            A_2 = get_entries(X, A_2_cds)
            A_T_2 = get_entries(X, A_T_2_cds)
            print "A_1:\n", A_1
            print "A_T_1:\n", A_T_1
            print "A_2:\n", A_2
            print "A_T_2:\n", A_T_2
        assert succeed == True

def test8():
    """
    Tests Q-solve on data generated from a run of Muller potential.

    min_R -log det R + Tr(RF)
          ------------------
         |D-ADA.T  I        |
    X =  |   I     R        |
         |            R  cI |
         |            cI  I |
          ------------------
    X is PSD
    """
    eps = 1e-4
    tol = 7e-2
    search_tol = 1e-2
    N_iter = 100
    dims = [6]
    np.set_printoptions(precision=2)
    np.seterr(divide='raise')
    np.seterr(over='raise')
    np.seterr(invalid='raise')
    import pdb, traceback, sys
    try:
        for dim in dims:
            block_dim = int(dim/3)

            # Generate initial data
            D = np.array([[0.00326556, 0.00196009],
                          [0.00196009, 0.00322879]])
            F = np.array([[2.62197238, 1.58163533],
                          [1.58163533, 2.58977211]])
            A = np.zeros((block_dim, block_dim))
            scale = 1./np.amax(np.linalg.eigh(D)[0])
            R = np.trace(D) + 2 * (1./scale) * np.trace(np.linalg.inv(D))
            Rs = [R]
            print "Rs: ", Rs
            print "scale: ", scale
            # Rescaling
            D *= scale
            print "D_scaled:\n", D
            As, bs, Cs, ds, Fs, gradFs, Gs, gradGs = \
                    Q_constraints(block_dim, A, F, D)
            g = GeneralSolver(dim, eps)
            def obj(X):
                return log_det_tr(scale*X, F)
            def grad_obj(X):
                return grad_log_det_tr(scale*X, F)
            g.save_constraints(obj, grad_obj, As, bs, Cs, ds,
                    Fs, gradFs, Gs, gradGs)
            (L, U, X, succeed) = g.solve(N_iter, tol, verbose=False,
                    interactive=False, debug=True, Rs=Rs)
            (D_ADA_T_cds, I_1_cds, I_2_cds, R_1_cds, R_2_cds) \
                    = Q_coords(block_dim)
            # Undo trace scaling
            if X != None:
                R_1  = scale*get_entries(X, R_1_cds)
                R_2  = scale*get_entries(X, R_2_cds)
                Q_1 = np.linalg.inv(R_1)
                Q_2 = np.linalg.inv(R_2)
                print "Q_1:\n", Q_1
                print "Q_2:\n", Q_2
                print "D_scaled:\n", D
                # Undo rescaling
                D *= (1./scale)
                if X != None:
                    print "nplinalg.norm(D, 2): ", np.linalg.norm(D, 2)
                    print "np.linalg.norm(Q_1,2): ", np.linalg.norm(Q_1,2)
                    print "np.linalg.norm(Q_2,2): ", np.linalg.norm(Q_2,2)
                    assert np.linalg.norm(Q_1,2) < 1.1*np.linalg.norm(D,2)
                    assert np.linalg.norm(Q_2,2) < 1.1*np.linalg.norm(D,2)
            assert succeed == True
    except:
        type, value, tb = sys.exc_info()
        traceback.print_exc()
        pdb.post_mortem(tb)

def test9():
    """
    Tests A-optimization on data generated from run of Muller potential.

    min_A Tr [ Q^{-1} ([C - B] A.T + A [C - B].T + A E A.T]

          --------------------
         | D-Q    A           |
    X =  | A.T  D^{-1}        |
         |              I   A |
         |             A.T  I |
          --------------------
    A mu == 0
    X is PSD

    If A is dim by dim, then this matrix is 4 * dim by 4 * dim.
    The solution to this problem is A = 0 when dim = 1.
    """
    eps = 1e-4
    tol = 2e-2
    search_tol = 1e-2
    N_iter = 100
    Rs = [10]
    np.set_printoptions(precision=2)
    dims = [8]
    import pdb, traceback, sys
    try:
        for dim in dims:
            block_dim = int(dim/4)

            # Generate random data
            D = np.array([[0.00326556, 0.00196009],
                          [0.00196009, 0.00322879]])
            Dinv = np.linalg.inv(D)
            Q = 0.9 * D
            Qinv = np.linalg.inv(Q)
            C = np.array([[202.83070879, -600.32796941],
                          [-601.76432584, 1781.07130791]])
            B = np.array([[208.27749525,  -597.11827148],
                          [ -612.99179464, 1771.25551671]])
            E = np.array([[205.80695137, -599.79918374],
                          [-599.79918374, 1782.52514543]])
            mu =  np.array([[-0.7010104, 1.29133034]])
            #mu =  np.array([[ 0.68616771, 0.02634688]])
            #mu =  np.array([[ 0.59087205,  0.03185492]])
            mu = np.reshape(mu, (block_dim, 1))

            scale = 1./np.amax(np.linalg.eigh(D)[0])
            # Rescaling
            D *= scale
            Q *= scale
            Dinv *= (1./scale)
            Qinv *= (1./scale)

            (D_Q_cds, Dinv_cds, I_1_cds, I_2_cds,
                A_1_cds, A_T_1_cds, A_2_cds, A_T_2_cds) \
                        = A_coords(block_dim)
            As, bs, Cs, ds, Fs, gradFs, Gs, gradGs = \
                    A_constraints(block_dim, D, Dinv, Q, mu)
            def obj(X):
                return A_dynamics(X, block_dim, C, B, E, Qinv)
            def grad_obj(X):
                return grad_A_dynamics(X, block_dim, C, B, E, Qinv)
            g = GeneralSolver(dim, eps)
            g.save_constraints(obj, grad_obj, As, bs, Cs, ds,
                    Fs, gradFs, Gs, gradGs)
            (L, U, X, succeed) = g.solve(N_iter, tol,
                    disp=True, interactive=False, debug=True,
                    verbose=False, Rs=Rs)
            # Undo trace scaling
            print "X\n", X
            if X != None:
                A_1 = get_entries(X, A_1_cds)
                A_T_1 = get_entries(X, A_T_1_cds)
                A_2 = get_entries(X, A_2_cds)
                A_T_2 = get_entries(X, A_T_2_cds)
                A = (1./4) * (A_1 + A_T_1 + A_2 + A_T_2)
                print "A_1:\n", A_1
                print "A_T_1:\n", A_T_1
                print "A_2:\n", A_2
                print "A_T_2:\n", A_T_2
                print "A:\n", A
            assert succeed == True
    except:
        type, value, tb = sys.exc_info()
        traceback.print_exc()
        pdb.post_mortem(tb)
