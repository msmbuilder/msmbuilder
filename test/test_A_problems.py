import numpy as np
from mixtape.mslds_solvers.A_problems import A_problem
from mixtape.mslds_solvers.sparse_sdp.utils import get_entries
from mixtape.mslds_solvers.sparse_sdp.utils import numerical_derivative
import time
import warnings

def test_objective():
    dims = [1, 2]
    N_rand = 10
    tol = 1e-3
    eps = 1e-4

    for dim in dims:
        a_prob = A_problem(dim)
        (D_Q_cds, Dinv_cds, A_cds, A_T_cds) = a_prob.coords()
        prob_dim = a_prob.scale * dim

        # Generate initial data
        D = np.eye(dim)
        Q = 0.5*np.eye(dim)
        Qinv = np.linalg.inv(Q)
        C = 2*np.eye(dim)
        B = np.eye(dim)
        E = np.eye(dim)
        def obj(X):
            return a_prob.objective(X, C, B, E, Qinv)
        def grad_obj(X):
            return a_prob.grad_objective(X, C, B, E, Qinv)
        for i in range(N_rand):
            X = np.random.rand(prob_dim, prob_dim)
            val = obj(X)
            grad = grad_obj(X)
            num_grad = numerical_derivative(obj, X, eps)
            diff = np.sum(np.abs(grad - num_grad))
            print "X:\n", X
            print "grad:\n", grad
            print "num_grad:\n", num_grad
            print "diff: ", diff
            assert diff < tol

def test_A_constraints():
    dims = [1, 2]
    N_rand = 10
    tol = 1e-3
    eps = 1e-4
    np.set_printoptions(precision=3)
    for dim in dims:
        a_prob = A_problem(dim)
        prob_dim = a_prob.scale * dim
        # Generate initial data
        D = np.eye(dim)
        Dinv = np.linalg.inv(D)
        Q = 0.5*np.eye(dim)
        mu = np.ones((dim, 1))
        As, bs, Cs, ds, Fs, gradFs, Gs, gradGs = \
                a_prob.constraints(D, Dinv, Q)
        N_rand = 10
        for (g, gradg) in zip(Gs, gradGs):
            for i in range(N_rand):
                X = np.random.rand(prob_dim, prob_dim)
                val = g(X)
                grad = gradg(X)
                print "grad:\n", grad
                num_grad = numerical_derivative(g, X, eps)
                print "num_grad:\n", num_grad
                assert np.sum(np.abs(grad - num_grad)) < tol

def test_A_solve_1():
    """
    Tests feasibility of A optimization.

    min_A Tr [ Q^{-1} ([C - B] A.T + A [C - B].T + A E A.T]

          -------------
         | D-Q    A    |
    X =  | A.T  D^{-1} |
          -------------
    X is PSD

    The solution to this problem is A = 0 when dim = 1.
    """
    tol = 1e-1
    search_tol = 2.
    N_iter = 100
    dims = [16, 32]
    scale = 10.
    for dim in dims:
        a_prob = A_problem(dim)

        # Generate random data
        D = .0204 * np.eye(dim)
        Q = .02 * np.eye(dim)
        C = 1225.025 * np.eye(dim)
        B = 1238.916 * np.eye(dim)
        E = 48.99 * np.eye(dim)

        # Call solver
        t_start = time.time()
        A = a_prob.solve(B, C, D, E, Q, tol=tol, search_tol=search_tol,
                            verbose=False)
        t_end = time.time()

        #print "A\n", A
        print "dim: ", dim
        print "total time: ", (t_end - t_start)
        assert np.linalg.norm(A,2) < 1

def test_A_solve_2():
    """
    Tests feasibility of A optimization.

    min_A Tr [ Q^{-1} ([C - B] A.T + A [C - B].T + A E A.T]

          -------------
         | D-Q    A    |
    X =  | A.T  D^{-1} |
          -------------
    X is PSD

    The solution to this problem is A = 0 when dim = 1.
    """
    dims = [4, 16]
    tol = 1e-1
    search_tol = 2.
    for dim in dims:
        a_prob = A_problem(dim)

        # Generate random data
        D = np.eye(dim)
        Q = 0.5 * np.eye(dim)
        C = 2 * np.eye(dim)
        B = np.eye(dim)
        E = np.eye(dim)

        # Call solver
        t_start = time.time()
        A = a_prob.solve(B, C, D, E, Q, tol=tol, search_tol=search_tol)
        t_end = time.time()

        print "total time: ", (t_end - t_start)
        assert np.linalg.norm(A,2) < 1

def test_A_solve_3():
    """
    Tests A-optimization on data generated from run of Muller potential.

    min_A Tr [ Q^{-1} ([C - B] A.T + A [C - B].T + A E A.T]

          -------------
         | D-Q    A    |
    X =  | A.T  D^{-1} |
          -------------
    X is PSD

    If A is dim by dim, then this matrix is 4 * dim by 4 * dim.
    The solution to this problem is A = 0 when dim = 1.
    """
    eps = 1e-4
    tol = 1e-1
    search_tol = 1
    N_iter = 100
    Rs = [10]
    np.set_printoptions(precision=2)
    dim = 2
    a_prob = A_problem(dim)

    # Generate random data
    D = np.array([[0.00326556, 0.00196009],
                  [0.00196009, 0.00322879]])
    Q = 0.9 * D
    C = np.array([[202.83070879, -600.32796941],
                  [-601.76432584, 1781.07130791]])
    B = np.array([[208.27749525,  -597.11827148],
                  [ -612.99179464, 1771.25551671]])
    E = np.array([[205.80695137, -599.79918374],
                  [-599.79918374, 1782.52514543]])
    # Call solver
    t_start = time.time()
    A = a_prob.solve(B, C, D, E, Q, tol=tol, search_tol=search_tol)
    t_end = time.time()

    print "A\n", A
    print "total time: ", (t_end - t_start)
    assert np.linalg.norm(A,2) < 1

def test_A_solve_plusmin():
    n_features = 1
    a_prob = A_problem(n_features)

    B = np.array([[1238.916]])
    C = np.array([[1225.025]])
    D = np.array([[.0204]])
    E = np.array([[48.99]])
    Q = np.array([[.02]])
    mu = np.array([[1.]])

    # Call solver
    t_start = time.time()
    A = a_prob.solve(B, C, D, E, Q)
    t_end = time.time()

    print "A\n", A
    print "total time: ", (t_end - t_start)
    assert np.linalg.norm(A,2) < 1
     
def test_A_solve_plusmin_2():
    n_features = 1
    a_prob = A_problem(n_features)

    B = np.array([[ 965.82431552]])
    C = np.array([[ 950.23843989]])
    D = np.array([[ 0.02430409]])
    E = np.array([[ 974.49540394]])
    F = np.array([[ 24.31867657]])
    Q = np.array([[ 0.02596519]])

    # Call solver
    t_start = time.time()
    A = a_prob.solve(B, C, D, E, Q)
    t_end = time.time()

    print "A\n", A
    print "total time: ", (t_end - t_start)
    assert np.linalg.norm(A,2) < 1

def test_A_solve_muller():
    n_features = 2
    a_prob = A_problem(n_features)
    B = np.array([[208.27749525,  -597.11827148],
                   [ -612.99179464, 1771.25551671]])

    C = np.array([[202.83070879, -600.32796941],
                   [-601.76432584, 1781.07130791]])

    D = np.array([[0.00326556, 0.00196009],
                   [0.00196009, 0.00322879]])

    E = np.array([[205.80695137, -599.79918374],
                  [-599.79918374, 1782.52514543]])
    Q = .9 * D

    # Call solver
    t_start = time.time()
    A = a_prob.solve(B, C, D, E, Q)
    t_end = time.time()

    print "A\n", A
    print "total time: ", (t_end - t_start)
    assert np.linalg.norm(A,2) < 1

def test_A_solve_muller_2():
    n_features = 2
    a_prob = A_problem(n_features)
    warnings.filterwarnings("ignore", category=DeprecationWarning)

    B = np.array([[  359.92406863,  -853.5934402 ],
                  [ -842.86780552,  2010.34907067]])

    C = np.array([[  361.80793384,  -850.60352492],
                  [ -851.82693628,  2002.62881727]])

    D = np.array([[ 0.00261965,  0.00152437],
                  [ 0.00152437,  0.00291518]])

    E = np.array([[  364.88271615,  -849.83206073],
                  [ -849.83206073,  2004.72145185]])
    Q = .9 * D

    # Call solver
    t_start = time.time()
    A = a_prob.solve(B, C, D, E, Q)
    t_end = time.time()

    print "A\n", A
    print "total time: ", (t_end - t_start)
    assert np.linalg.norm(A,2) < 1

def test_A_solve_muller_3():
    n_features = 2
    a_prob = A_problem(n_features)
    warnings.filterwarnings("ignore", category=DeprecationWarning)

    Q = np.array([[ 0.00268512, -0.00030655],
                 [-0.00030655,  0.002112  ]])
    B = np.array([[ 269.81124024,   15.28704689],
                    [  16.32464053,    0.99806799]])
    C = np.array([[ 266.55743817,   16.13788517],
                    [  16.01112828,    0.96934361]])
    D = np.array([[ 0.00246003, -0.00017837],
                    [-0.00017837,  0.00190514]])
    E = np.array([[ 267.86405002,   15.94161187],
                    [  15.94161187,    2.47997446]])
    F = np.array([[ 1.97090458, -0.15635765],
                    [-0.15635765,  1.50541836]])

    # Call solver
    t_start = time.time()
    A = a_prob.solve(B, C, D, E, Q)
    t_end = time.time()

    print "A\n", A
    print "total time: ", (t_end - t_start)
    assert np.linalg.norm(A,2) < 1

def test_A_solve_muller_4():
    n_features = 2
    a_prob = A_problem(n_features)
    warnings.filterwarnings("ignore", category=DeprecationWarning)

    B = (np.array([[  47.35392822,  -87.69193367],
                [ -83.75658227,  155.95421092]]))
    C = (np.array([[  47.72109794,  -84.70032366],
                [ -86.21727938,  153.02731464]]))
    D = (np.array([[ 0.34993938, -0.3077952 ],
                [-0.3077952 ,  0.854263  ]]))
    E = (np.array([[  48.90734354, -86.23552793],
                [ -86.23552793, 153.58913243]]))
    Q = (np.array([[ 0.00712922, -0.00164496],
                [-0.00164496, 0.01020176]]))

    # Call solver
    t_start = time.time()
    A = a_prob.solve(B, C, D, E, Q)
    t_end = time.time()

    print "A\n", A
    print "total time: ", (t_end - t_start)
    assert np.linalg.norm(A,2) < 1

def test_A_solve_alanine():
    #Auto-generated test case from failing run of
    #A-solve:
    import pickle
    n_features = 66 
    a_prob = A_problem(n_features)
    tol=1e-1

    B = pickle.load(open("B_alanine_1.p", "r"))
    C = pickle.load(open("C_alanine_1.p", "r"))
    D = pickle.load(open("D_alanine_1.p", "r"))
    E = pickle.load(open("E_alanine_1.p", "r"))
    Q = pickle.load(open("Q_alanine_1.p", "r"))

    # Call solver
    t_start = time.time()
    A = a_prob.solve(B, C, D, E, Q, tol=tol)
    t_end = time.time()

    print "total time: ", (t_end - t_start)
    assert np.linalg.norm(A,2) < 1
