import sys
sys.path.append("..")
from bounded_trace_sdp_solver import BoundedTraceSolver
from objectives import neg_sum_squares, grad_neg_sum_squares
from constraints import *
from penalties import *
import time
import scipy
import numpy as np
"""
Tests for Hazan's core algorithm.

@author: Bharath Ramsundar
@email: bharath.ramsundar@gmail.com

TODOs:
    -) Remove m, n from the test cases.
"""

def test1():
    """
    Test bounded trace solver on function f(x)  = -\sum_k x_kk^2
    defined above.

    Now do a dummy optimization problem. The
    problem we consider is

        max - sum_k x_k^2
        subject to
            Tr(X) = 1

    The optimal solution is -1/n, where
    n is the dimension.
    """
    eps = 1e-3
    N_iter = 50
    # dimension of square matrix X
    dims = [16]
    for dim in dims:
        print("dim = %d" % dim)
        b = BoundedTraceSolver(neg_sum_squares, grad_neg_sum_squares, dim)
        X = b.solve(N_iter)
        fX = neg_sum_squares(X)
        print("\tTr(X) = %f" % np.trace(X))
        print("\tf(X) = %f" % fX)
        print("\tf* = %f" % (-1./dim))
        print("\t|f(X) - f*| = %f" % (np.abs(fX - (-1./dim))))
        assert np.abs(fX - (-1./dim)) < eps

def test2a():
    """
    Check equality constraints for log_sum_exp constraints
    """
    eps = 1e-3
    N_iter = 50
    dim, As, bs, Cs, ds, Fs, gradFs, Gs, gradGs = \
           simple_equality_constraint()
    M = compute_scale(len(As), len(Cs), len(Fs), len(Gs), eps)
    def f(X):
        return log_sum_exp_penalty(X, M, As, bs, Cs, ds, Fs, Gs)
    def gradf(X):
        return log_sum_exp_grad_penalty(X, M, As,
                    bs, Cs, ds, Fs, gradFs, Gs, gradGs)
    B = BoundedTraceSolver(f, gradf, dim)
    X, elapsed  = run_experiment(B, N_iter)
    succeed = not (f(X) < -eps)
    print "\tComputation Time (s): ", elapsed
    assert succeed == True

def test2b():
    """
    BROKEN: Check equality constraints for neg_max constraints
    TODO: Fix this test
    """
    N_iter = 50
    assert True == False
    m, n, M, dim, eps, As, bs, Cs, ds, Fs, gradFs, Gs, gradGs = \
           simple_equality_constraint(N_iter)
    def f(X):
        return neg_max_penalty(X, M, As, bs, Cs, ds, Fs, Gs)
    def gradf(X):
        return neg_max_grad_penalty(X, M, As, bs, Cs, ds, Fs, gradFs,
                Gs, gradGs, eps)
    run_experiment(f, gradf, dim, N_iter)

def test3a():
    """
    Check equality and inequality constraints for log_sum_exp penalty
    """
    eps = 1e-3
    N_iter = 50
    dim, As, bs, Cs, ds, Fs, gradFs, Gs, gradGs = \
            simple_equality_and_inequality_constraint()
    M = compute_scale(len(As), len(Cs), len(Fs), len(Gs), eps)
    def f(X):
        return log_sum_exp_penalty(X, M, As, bs, Cs, ds, Fs, Gs)
    def gradf(X):
        return log_sum_exp_grad_penalty(X, M, As, bs, Cs, ds,
                Fs, gradFs, Gs, gradGs)
    B = BoundedTraceSolver(f, gradf, dim)
    X, elapsed  = run_experiment(B, N_iter)
    succeed = not (f(X) < -eps)
    print "\tComputation Time (s): ", elapsed
    assert succeed == True

def test3b():
    """
    BROKEN: Check equality and inequality constraints for neg_max penalty
    TODO: Fix this test
    """
    assert True == False
    N_iter = 50
    m, n, M, As, bs, Cs, ds, dim, eps, Fs, gradFs, Gs, gradGs = \
            simple_constraint(N_iter)
    def f(X):
        return neg_max_penalty(X, M, As, bs, Cs, ds, Fs, Gs)
    def gradf(X):
        return neg_max_grad_penalty(X, M, As, bs, Cs, ds, Fs, gradFs,
                Gs, gradGs, eps)
    X, fX, SUCCEED = run_experiment(f, gradf, dim, N_iter)

def test3c():
    """
    BROKEN: Check equality and inequality constraints for neg_max penalty
    with log_sum_exp gradients.
    TODO: Fix this test
    """
    assert True == False
    N_iter = 50
    m, n, M, As, bs, Cs, ds, dim, eps, Fs, gradFs, Gs, gradGs = \
            simple_constraint(N_iter)
    def f(X):
        return neg_max_penalty(X, M, As, bs, Cs, ds, Fs, Gs)
    def gradf(X):
        return log_sum_exp_grad_penalty(X, M,
                As, bs, Cs, ds, Fs, gradFs, Gs, gradGs, eps)
    X, fX, SUCCEED = run_experiment(f, gradf, dim, N_iter)

def test4a():
    """
    BROKEN: Check quadratic inequality constraints for neg_max penalty
    and gradients.
    TODO: Fix this test
    """
    assert True == False
    N_iter = 50
    dim, M, As, bs, Cs, ds, Fs, gradFs, Gs, gradGs, eps = \
            quadratic_inequality(N_iter)
    def f(X):
        return neg_max_penalty(X, M, As, bs, Cs, ds, Fs, Gs)
    def gradf(X):
        return neg_max_grad_penalty(X, M,
                As, bs, Cs, ds, Fs, gradFs, Gs, gradGs, eps)
    X, fX, SUCCEED = run_experiment(f, gradf, dim, N_iter)

def test4b():
    """
    Check quadratic inequality for log_sum_exp penalty and gradients.
    """
    eps = 1e-3
    N_iter = 50
    dim, As, bs, Cs, ds, Fs, gradFs, Gs, gradGs = \
            quadratic_inequality()
    M = compute_scale(len(As), len(Cs), len(Fs), len(Gs), eps)
    def f(X):
        return log_sum_exp_penalty(X, M, As, bs, Cs, ds, Fs, Gs)
    def gradf(X):
        return log_sum_exp_grad_penalty(X, M,
                As, bs, Cs, ds, Fs, gradFs, Gs, gradGs)
    B = BoundedTraceSolver(f, gradf, dim)
    X, elapsed  = run_experiment(B, N_iter)
    succeed = not (f(X) < -eps)
    print "\tComputation Time (s): ", elapsed
    assert succeed == True

def test5a():
    """
    BROKEN: quadratic equality constraints for neg_max penalty
    and gradients.
    """
    assert True == False
    N_iter = 50
    dim, M, As, bs, Cs, ds, Fs, gradFs, Gs, gradGs, eps = \
            quadratic_equality(N_iter)
    def f(X):
        return neg_max_penalty(X, M, As, bs, Cs, ds, Fs, Gs)
    def gradf(X):
        return neg_max_grad_penalty(X, M,
                As, bs, Cs, ds, Fs, gradFs, Gs, gradGs, eps)
    X, fX, SUCCEED = run_experiment(f, gradf, dim, N_iter)

def test6a():
    """
    Stress test inequality constraints for log_sum_exp penalty.
    """
    eps = 1e-3
    dims = [4,16]
    N_iter = 50
    for dim in dims:
        As, bs, Cs, ds, Fs, gradFs, Gs, gradGs = \
                stress_inequalities(dim)
        M = compute_scale(len(As), len(Cs), len(Fs), len(Gs), eps)
        def gen_f(M, As, bs, Cs, ds, Fs, Gs):
            def f(X):
                return log_sum_exp_penalty(X, M, As, bs, Cs, ds, Fs, Gs)
            return f
        f = gen_f(M, As, bs, Cs, ds, Fs, Gs)
        def gen_gradf(M, As, bs, Cs, ds, Fs, gradFs, Gs, gradGs):
            def gradf(X):
                return log_sum_exp_grad_penalty(X, M,
                        As, bs, Cs, ds, Fs, gradFs, Gs, gradGs)
            return gradf
        gradf = gen_gradf(M, As, bs, Cs, ds, Fs, gradFs, Gs, gradGs)
        B = BoundedTraceSolver(f, gradf, dim)
        X, elapsed  = run_experiment(B, N_iter)
        succeed = not (f(X) < -eps)
        print "\tComputation Time (s): ", elapsed
        assert succeed == True

def test6b():
    """
    BROKEN: Stress test inequality constraints for neg_max_sum penatly
    and log_sum_exp gradient.
    """
    assert True == False
    dims = [4,16]
    N_iter = 50
    for dim in dims:
        m, n, M, As, bs, Cs, ds, eps, Fs, gradFs, Gs, gradGs = \
                stress_inequalities(dim, N_iter)
        def f(X):
            return neg_max_penalty(X, M, As, bs, Cs, ds, Fs, Gs)
        def gradf(X):
            return log_sum_exp_grad_penalty(X, M,
                    As, bs, Cs, ds, Fs, gradFs, Gs, gradGs, eps)
        run_experiment(f, gradf, dim, N_iter)

def test6c():
    """
    BROKEN: Stress test inequality constraints for neg_max_penalty
    """
    assert True == False
    dims = [4,16]
    N_iter = 50
    for dim in dims:
        m, n, M, As, bs, Cs, ds, eps, Fs, gradFs, Gs, gradGs = \
                stress_inequalities(dim, N_iter)
        def f(X):
            return neg_max_penalty(X, M, As, bs, Cs, ds, Fs, Gs)
        def gradf(X):
            return neg_max_grad_penalty(X, M, As, bs, Cs, ds,
                    Fs, gradFs, Gs, gradGs, eps)
        run_experiment(f, gradf, dim, N_iter)

def test7a():
    """
    Stress test equality constraints for log_sum_exp_penalty
    """
    eps = 1e-3
    dims = [4,16]
    N_iter = 50
    for dim in dims:
        As, bs, Cs, ds, Fs, gradFs, Gs, gradGs = \
                stress_equalities(dim)
        M = compute_scale(len(As), len(Cs), len(Fs), len(Gs), eps)
        def gen_f(M, As, bs, Cs, ds, Fs, Gs):
            def f(X):
                return log_sum_exp_penalty(X, M, As, bs, Cs, ds, Fs, Gs)
            return f
        f = gen_f(M, As, bs, Cs, ds, Fs, Gs)
        def gen_gradf(M, As, bs, Cs, ds, Fs, gradFs, Gs, gradGs):
            def gradf(X):
                return log_sum_exp_grad_penalty(X, M,
                        As, bs, Cs, ds, Fs, gradFs, Gs, gradGs)
            return gradf
        gradf = gen_gradf(M, As, bs, Cs, ds, Fs, gradFs, Gs, gradGs)
        B = BoundedTraceSolver(f, gradf, dim)
        X, elapsed  = run_experiment(B, N_iter)
        succeed = not (f(X) < -eps)
        print "\tComputation Time (s): ", elapsed
        assert succeed == True

def test7b():
    """
    BROKEN: Stress test equality constraints for neg_max_penalty
    """
    assert True == False
    dims = [4,16]
    N_iter = 50
    for dim in dims:
        m, n, M, As, bs, Cs, ds, eps, Fs, gradFs, Gs, gradGs = \
                stress_equalities(dim, N_iter)
        def f(X):
            return neg_max_penalty(X, M, As, bs, Cs, ds, Fs, Gs)
        def gradf(X):
            return log_sum_exp_grad_penalty(X, M, As, bs, Cs, ds,
                    Fs, gradFs, Gs, gradGs, eps)
        run_experiment(f, gradf, dim, N_iter)

def test7c():
    """
    BROKEN: Stress test equality constraints for neg_max_penalty
    """
    assert True == False
    dims = [4,16]
    N_iter = 50
    for dim in dims:
        m, n, M, As, bs, Cs, ds, eps, Fs, gradFs, Gs, gradGs = \
                stress_equalities(dim, N_iter)
        def f(X):
            return neg_max_penalty(X, M, As, bs, Cs, ds, Fs, Gs)
        def gradf(X):
            return neg_max_grad_penalty(X, M, As, bs, Cs, ds,
                    Fs, gradFs, Gs, gradGs, eps)
        run_experiment(f, gradf, dim, N_iter)

def stress_inequalies_and_equalities(dim, N_iter):
    """
    Stress test the bounded trace solver for both equalities and
    inequalities.

    With As and bs as below, we specify the problem

    max neg_max_penalty(X)
    subject to
        x_ij == 0, i != j
        x11
        Tr(X) = x_11 + x_22 + ... + x_nn == 1

    The optimal solution should equal a diagonal matrix with zero entries
    for the first n-1 diagonal elements, but a 1 for the diagonal element.

    """
    eps = 1./N_iter
    np.set_printoptions(precision=2)
    m = dim - 2
    n = dim**2 - dim
    As = []
    M = compute_scale(m, n, eps)
    for j in range(1,dim-1):
        Aj = np.zeros((dim,dim))
        Aj[j,j] = 1
        As.append(Aj)
    bs = []
    for j in range(1,dim-1):
        bj = 1./N_iter
        bs.append(bj)
    Cs = []
    for i in range(dim):
        for j in range(dim):
            if i != j:
                Ci = np.zeros((dim,dim))
                Ci[i,j] = 1
                Cs.append(Ci)
    ds = []
    for i in range(dim):
        for j in range(dim):
            if i != j:
                dij = 0.
                ds.append(dij)
    Fs = []
    gradFs = []
    Gs = []
    gradGs = []
    return m, n, M, As, bs, Cs, ds, eps, Fs, gradFs, Gs, gradGs

def test8a():
    """
    Stress test equality constraints for log_sum_exp_penalty
    """
    dims = [4,16]
    N_iter = 200
    for dim in dims:
        m, n, M, As, bs, Cs, ds, eps, Fs, gradFs, Gs, gradGs = \
               stress_inequalies_and_equalities(dim, N_iter)
        def f(X):
            return log_sum_exp_penalty(X, M, As, bs, Cs, ds, Fs, Gs)
        def gradf(X):
            return log_sum_exp_grad_penalty(X, M,
                        As, bs, Cs, ds, Fs, gradFs, Gs, gradGs, eps)
        X, fX, SUCCEED = run_experiment(f, gradf, dim, N_iter)

def test8b():
    """
    Stress test equality constraints for neg_max_penalty
    """
    dims = [4, 16]
    N_iter = 50
    for dim in dims:
        m, n, M, As, bs, Cs, ds, eps, Fs, gradFs, Gs, gradGs = \
               stress_inequalies_and_equalities(dim, N_iter)
        def f(X):
            return neg_max_penalty(X, M, As, bs, Cs, ds, Fs, Gs)
        def gradf(X):
            return log_sum_exp_grad_penalty(X, M, As, bs, Cs, ds,
                    Fs, gradFs, Gs, gradGs, eps)
        X, fX, SUCCEED = run_experiment(f, gradf, dim, N_iter)

def test8c():
    """
    Stress test equality constraints for neg_max_penalty
    """
    dims = [4, 16]
    N_iter = 50
    for dim in dims:
        m, n, M, As, bs, Cs, ds, eps, Fs, gradFs, Gs, gradGs = \
               stress_inequalies_and_equalities(dim, N_iter)
        def f(X):
            return neg_max_penalty(X, M, As, bs, Cs, ds, Fs, Gs)
        def gradf(X):
            return neg_max_grad_penalty(X, M,
                        As, bs, Cs, ds, Fs, gradFs, Gs, gradGs, eps)
        X, fX, SUCCEED = run_experiment(f, gradf, dim, N_iter)

def batch_equality(A, dim, N_iter):
    """
    Check that the bounded trace implementation can handle batch
    equality constraints in a matrix.

    We specify the problem

    feasibility(X)
    subject to
        [[ B   , A],
         [ A.T , D]]  is PSD, where B, D are arbitrary, A given.

        Tr(X) = Tr(B) + Tr(D) == 1
    """
    m = 0
    As = []
    bs = []
    n = 0
    Cs = []
    ds = []
    p = 0
    Fs = []
    gradFs = []
    q = 1
    block_dim = int(dim/2)
    # TODO: Swap this out with generalized constraint
    def g(X):
        c1 = np.sum(np.abs(X[:block_dim,block_dim:] - A))
        c2 = np.sum(np.abs(X[block_dim:,:block_dim] - A.T))
        return c1 + c2
    def gradg(X):
        # TODO: Maybe speed this up and avoid allocating new matrix
        #       of zeros every gradient computation.
        # Upper right
        grad1 = np.sign(X[:block_dim,block_dim:] - A)
        # Lower left
        grad2 = np.sign(X[block_dim:,:block_dim] - A.T)

        grad = np.zeros((dim, dim))
        grad[:block_dim,block_dim:] = grad1
        grad[block_dim:,:block_dim] = grad2
        # Not sure if this is right...
        return grad

    Gs = [g]
    gradGs = [gradg]
    eps = 1./N_iter
    M = compute_scale_full(m, n, p, q, eps)
    return M, As, bs, Cs, ds, Fs, gradFs, Gs, gradGs, eps

def test9a():
    """
    Test block equality constraints.
    """
    dims = [4, 16]
    N_iter = 200
    #alphas = 0.01 * np.ones(N_iter)
    #alphas = 5 * [1./(j+1) for j in range(N_iter)]
    alphas = None
    DEBUG = False
    for dim in dims:
        A = (1./(2*dim)) * np.eye(int(dim/2))
        M, As, bs, Cs, ds, Fs, gradFs, Gs, gradGs, eps = \
                batch_equality(A, dim, N_iter)
        def f(X):
            return neg_max_penalty(X, M, As, bs, Cs, ds, Fs, Gs)
        def gradf(X):
            return neg_max_grad_penalty(X, M,
                        As, bs, Cs, ds, Fs, gradFs, Gs, gradGs, eps)
        def z(X):
            lambda_max = np.amax(np.linalg.eigh(gradf(X))[0])
            return lambda_max
        def frob(X):
            tr = np.trace(np.dot(X, gradf(X)))
            return tr
        def w_dual(X):
            zX = z(X)
            fX = f(X)
            tr = frob(X)
            return zX + fX - tr
        def grad_update(X):
            G = gradf(X)
            wj, vj = scipy.sparse.linalg.eigsh(G, k=1, which='LA',
                        tol=0.)
            return np.outer(vj, vj)
        X, fX, SUCCEED = run_experiment(f, gradf, dim, N_iter,
                alphas=alphas,DEBUG=DEBUG)
        g = Gs[0]
        gradg = gradGs[0]

def run_experiment(B, N_iter, disp=True, debug=False):
    start = time.clock()
    X = B.solve(N_iter, disp=disp, debug=debug)
    elapsed = (time.clock() - start)
    return X, elapsed
