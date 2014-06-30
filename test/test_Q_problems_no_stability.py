import numpy as np
from mixtape.mslds_solvers.Q_problems_no_stability import Q_problem_no_stability
from mixtape.mslds_solvers.sparse_sdp.utils import get_entries
from mixtape.mslds_solvers.sparse_sdp.utils import numerical_derivative
import time
import warnings

def test_log_det():
    dims = [1]
    N_rand = 10
    tol = 1e-1
    eps = 1e-4
    for dim in dims:
        q_prob = Q_problem_no_stability(dim)
        (R_cds) = q_prob.coords()
        prob_dim = q_prob.scale * dim

        # Generate initial data
        F = np.random.rand(dim, dim)
        G = np.random.rand()
        def obj(X):
            return q_prob.objective(X, F, G)
        def grad_obj(X):
            return q_prob.grad_objective(X, F, G)
        for i in range(N_rand):
            X = np.random.rand(prob_dim, prob_dim)
            R = get_entries(X, R_cds)
            if (np.linalg.det(R) <= 0):
                continue
            val = obj(X)
            grad = grad_obj(X)
            num_grad = numerical_derivative(obj, X, eps)
            diff = np.sum(np.abs(grad - num_grad))
            if diff >= tol:
                print "grad:\n", grad
                print "num_grad:\n", num_grad
                print "diff: ", diff
            assert diff < tol

def test_Q_constraints():
    dims = [1, 2]
    N_rand = 10
    eps = 1e-4
    tol = 1e-3
    for dim in dims:
        q_prob = Q_problem_no_stability(dim)
        prob_dim = q_prob.scale * dim

        # Generate initial data
        D = np.eye(dim)
        Dinv = np.linalg.inv(D)
        B = np.eye(dim)
        A = 0.5 * np.eye(dim)
        As, bs, Cs, ds, Fs, gradFs, Gs, gradGs = \
                q_prob.constraints(A, B, D)
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

def test_Q_solve_1():
    """
    Tests that feasibility of Q optimization runs.

    min_Q -log det R + Tr(RF)
         ---
    X = | R |
         ---
    X is PSD
    """
    tol = 1e-2
    search_tol = 5e-2
    N_iter = 50
    dims = [1]

    for dim in dims:

        q_prob = Q_problem_no_stability(dim)
        prob_dim = q_prob.scale * dim

        # Generate initial data
        D = np.eye(dim) 
        F = np.eye(dim)
        A = 0.5*(1./dim) * np.eye(dim)
        G = 1.

        # Call solver
        t_start = time.time()
        Q = q_prob.solve(A, D, F, G, tol=tol, search_tol=search_tol)
        t_end = time.time()

        print "Q\n", Q
        print "dim: ", dim
        print "norm(Q, 2): ", np.linalg.norm(Q, 2)
        print "total time: ", (t_end - t_start)

def test_Q_solve_2():
    """
    Tests feasibility Q optimization with realistic values for F,
    D, A from the 1-d 2-well toy system.


    min_Q -log det R + Tr(RF)
         ---
    X = | R |
         ---
    X is PSD
    """
    tol = 1e-2
    search_tol = 1.
    N_iter = 150
    dims = [16]
    for dim in dims:
        q_prob = Q_problem_no_stability(dim)
        prob_dim = q_prob.scale * dim

        # Generate initial data
        D = .0204 * np.eye(dim)
        F = 25.47 * np.eye(dim)
        A = np.zeros(dim)
        G = 1.

        # Call solver
        t_start = time.time()
        Q = q_prob.solve(A, D, F, G, tol=tol, search_tol=search_tol)
        t_end = time.time()

        print "dim: ", dim
        print "total time: ", (t_end - t_start)
        print "norm(Q, 2): ", np.linalg.norm(Q, 2)

def test_Q_solve_3():
    """
    Tests Q-solve on data generated from a run of Muller potential.

    min_R -log det R + Tr(RF)
          -----------
         |D-ADA.T  cI |
    X =  |   cI     R |
          -----------
    X is PSD
    """
    tol = 1e-2
    search_tol = 1
    N_iter = 100
    dims = [2]
    np.seterr(divide='raise')
    np.seterr(over='raise')
    np.seterr(invalid='raise')
    for dim in dims:
        q_prob = Q_problem_no_stability(dim)
        prob_dim = q_prob.scale * dim

        # Generate initial data
        D = np.array([[0.00326556, 0.00196009],
                      [0.00196009, 0.00322879]])
        F = np.array([[2.62197238, 1.58163533],
                      [1.58163533, 2.58977211]])
        A = np.zeros((dim, dim))
        G = 1.

        # Call solver
        t_start = time.time()
        Q = q_prob.solve(A, D, F, G, tol=tol, search_tol=search_tol)
        t_end = time.time()

        print "D:\n", D
        print "Q:\n", Q
        print "norm(Q, 2): ", np.linalg.norm(Q, 2)

def test_Q_solve_plusmin():
    n_features = 1 
    q_prob = Q_problem_no_stability(n_features)
    np.set_printoptions(precision=4)

    A = np.array([[.0]])
    D = np.array([[.0204]])
    F = np.array([[25.47]])
    G = 1.

    # Call solver
    t_start = time.time()
    Q = q_prob.solve(A, D, F, G)
    t_end = time.time()

    print "D:\n", D
    print "Q:\n", Q
    print "total time: ", (t_end - t_start)
    print "norm(Q, 2): ", np.linalg.norm(Q, 2)


def test_Q_solve_muller():
    n_features = 2 
    q_prob = Q_problem_no_stability(n_features)
    np.set_printoptions(precision=4)

    A = np.zeros((n_features, n_features))
    D = np.array([[0.00326556, 0.00196009],
                   [0.00196009, 0.00322879]])
    F = np.array([[2.62197238, 1.58163533],
                  [1.58163533, 2.58977211]])
    G = 1.

    # Call solver
    t_start = time.time()
    Q = q_prob.solve(A, D, F, G)
    t_end = time.time()

    print "D:\n", D
    print "Q:\n", Q
    print "norm(Q, 2): ", np.linalg.norm(Q, 2)
    print "total time: ", (t_end - t_start)

def test_Q_solve_muller_2():
    n_features = 2 
    q_prob = Q_problem_no_stability(n_features)
    np.set_printoptions(precision=4)
    warnings.filterwarnings("ignore", category=DeprecationWarning)

    A = np.zeros((n_features, n_features))
    D = np.array([[ 0.00261965,  0.00152437],
                  [ 0.00152437,  0.00291518]])
    F = np.array([[ 2.72226628,  1.60237858],
                  [ 1.60237858,  3.0191094 ]])
    G = 1.

    # Call solver
    t_start = time.time()
    Q = q_prob.solve(A, D, F, G)
    t_end = time.time()

    print "D:\n", D
    print "Q:\n", Q
    print "norm(Q, 2): ", np.linalg.norm(Q, 2)
    print "total time: ", (t_end - t_start)

def test_Q_solve_muller_3():
    n_features = 2 
    q_prob = Q_problem_no_stability(n_features)
    np.set_printoptions(precision=4)
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    from numpy import array, float32

    A = np.zeros((n_features, n_features))
    D = (array([[ 0.00515675,  0.00027678],
                [ 0.00027678,  0.0092519 ]], dtype=float32))
    F = (array([[ 5.79813337, -2.13557243],
                [-2.13554192, -6.50420761]], dtype=float32))
    G = 1.

    # Call solver
    t_start = time.time()
    Q = q_prob.solve(A, D, F, G)
    t_end = time.time()

    print "D:\n", D
    print "Q:\n", Q
    print "norm(Q, 2): ", np.linalg.norm(Q, 2)
    print "total time: ", (t_end - t_start)
