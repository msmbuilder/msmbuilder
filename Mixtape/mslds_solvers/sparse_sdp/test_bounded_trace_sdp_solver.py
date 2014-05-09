from bounded_trace_sdp_solver import BoundedTraceSolver
from objectives import neg_sum_squares, grad_neg_sum_squares
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
    N_iter = 50
    # dimension of square matrix X
    #dims = [1,4,16,64]
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
        print("\tError Tolerance 1/%d = %f" % (N_iter, 1./N_iter))
        assert np.abs(fX - (-1./dim)) < 1./N_iter

def simple_equality_constraint(N_iter):
    """
    Check that the bounded trace implementation can handle low dimensional
    equality type constraints for the given penalty function.

    With As and bs as below, we specify the problem

        feasibility(X)
        subject to
          x_11 + 2 x_22 == 1.5
          Tr(X) = x_11 + x_22 == 1.

    We should find penalty(X) >= -eps, and that the above constraints have
    a solution
    """
    dim = 2
    eps = 1./N_iter
    As = []
    bs = []
    Cs = [np.array([[ 1.,  0.],
                    [ 0.,  2.]])]
    ds = [1.5]
    Fs = []
    gradFs = []
    Gs = []
    gradGs = []
    return dim, eps, As, bs, Cs, ds, Fs, gradFs, Gs, gradGs

def test2a():
    """
    Check equality constraints for log_sum_exp constraints
    """
    N_iter = 50
    dim, eps, As, bs, Cs, ds, Fs, gradFs, Gs, gradGs = \
           simple_equality_constraint(N_iter)
    def f(X):
        return log_sum_exp_penalty(X, M, As, bs, Cs, ds, Fs, Gs)
    def gradf(X):
        return log_sum_exp_grad_penalty(X, M, As,
                    bs, Cs, ds, Fs, gradFs, Gs, gradGs, eps)
    run_experiment(f, gradf, dim, N_iter)

def test2b():
    """
    Check equality constraints for neg_max constraints
    """
    N_iter = 50
    m, n, M, dim, eps, As, bs, Cs, ds, Fs, gradFs, Gs, gradGs = \
           simple_equality_constraint(N_iter)
    def f(X):
        return neg_max_penalty(X, M, As, bs, Cs, ds, Fs, Gs)
    def gradf(X):
        return neg_max_grad_penalty(X, M, As, bs, Cs, ds, Fs, gradFs,
                Gs, gradGs, eps)
    run_experiment(f, gradf, dim, N_iter)

def simple_constraint(N_iter):
    """
    Check that the bounded trace implementation can handle low-dimensional
    equality and inequality type constraints for given penalty.

    With As and bs as below, we specify the problem

        feasbility(X)
        subject to
            x_11 + 2 x_22 <= 1
            x_11 + 2 x_22 + 2 x_33 == 5/3
            Tr(X) = x_11 + x_22 + x_33 == 1
    """
    m = 1
    n = 1
    dim = 3
    eps = 1./N_iter
    M = compute_scale(m, n, eps)
    As = [np.array([[ 1., 0., 0.],
                    [ 0., 2., 0.],
                    [ 0., 0., 0.]])]
    bs = [1.]
    Cs = [np.array([[ 1.,  0., 0.],
                    [ 0.,  2., 0.],
                    [ 0.,  0., 2.]])]
    ds = [5./3]
    Fs = []
    gradFs = []
    Gs = []
    gradGs = []
    return m, n, M, As, bs, Cs, ds, dim, eps, Fs, gradFs, Gs, gradGs

def test3a():
    """
    Check equality and inequality constraints for log_sum_exp penalty
    """
    N_iter = 50
    m, n, M, As, bs, Cs, ds, dim, eps, Fs, gradFs, Gs, gradGs = \
            simple_constraint(N_iter)
    def f(X):
        return log_sum_exp_penalty(X, M, As, bs, Cs, ds, Fs, Gs)
    def gradf(X):
        return log_sum_exp_grad_penalty(X, M, As, bs, Cs, ds,
                Fs, gradFs, Gs, gradGs, eps)
    X, fX, SUCCEED = run_experiment(f, gradf, dim, N_iter)

def test3b():
    """
    Check equality and inequality constraints for neg_max penalty
    """
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
    Check equality and inequality constraints for neg_max penalty
    with log_sum_exp gradients.
    """
    N_iter = 50
    m, n, M, As, bs, Cs, ds, dim, eps, Fs, gradFs, Gs, gradGs = \
            simple_constraint(N_iter)
    def f(X):
        return neg_max_penalty(X, M, As, bs, Cs, ds, Fs, Gs)
    def gradf(X):
        return log_sum_exp_grad_penalty(X, M,
                As, bs, Cs, ds, Fs, gradFs, Gs, gradGs, eps)
    X, fX, SUCCEED = run_experiment(f, gradf, dim, N_iter)

def quadratic_inequality(N_iter):
    """
    Check that the bounded trace implementation can handle low-dimensional
    quadratic inequality.

    We specify the problem

        max penalty(X)
        subject to
            x_11^2 + x_22^2 <= .5
            Tr(X) = x_11 + x_22 == 1
    """
    dim = 2
    eps = 1./N_iter
    m = 0
    As = []
    bs = []
    n = 0
    Cs = []
    ds = []
    p = 1
    def f(X):
        return X[0,0]**2 + X[1,1]**2 - 0.5
    def gradf(X):
        grad = np.zeros(np.shape(X))
        grad[0,0] = 2 * X[0,0]
        grad[1,1] = 2 * X[1,1]
        return grad
    Fs = [f]
    gradFs = [gradf]
    q = 0
    Gs = []
    gradGs = []
    M = compute_scale_full(m, n, p, q, eps)
    return dim, M, As, bs, Cs, ds, Fs, gradFs, Gs, gradGs, eps

def test4a():
    """
    Check quadratic inequality constraints for neg_max penalty
    and gradients.
    """
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
    N_iter = 50
    dim, M, As, bs, Cs, ds, Fs, gradFs, Gs, gradGs, eps = \
            quadratic_inequality(N_iter)
    def f(X):
        return log_sum_exp_penalty(X, M, As, bs, Cs, ds, Fs, Gs)
    def gradf(X):
        return log_sum_exp_grad_penalty(X, M,
                As, bs, Cs, ds, Fs, gradFs, Gs, gradGs, eps)
    X, fX, SUCCEED = run_experiment(f, gradf, dim, N_iter)

def quadratic_equality(N_iter):
    """
    Check that the bounded trace implementation can handle
    low-dimensional quadratic equalities

    We specify the problem

        feasibility(X)
        subject to
            x_11^2 + x_22^2 = 0.5
            Tr(X) = x_11 + x_22 == 1
    """
    dim = 2
    eps = 1./N_iter
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
    def g(X):
        return X[0,0]**2 + X[1,1]**2 - 0.5
    def gradg(X):
        grad = np.zeros(np.shape(X))
        grad[0,0] = 2 * X[0,0]
        grad[1,1] = 2 * X[1,1]
        return grad
    Gs = [g]
    gradGs = [gradg]
    M = compute_scale_full(m, n, p, q, eps)
    return dim, M, As, bs, Cs, ds, Fs, gradFs, Gs, gradGs, eps

def test5a():
    """
    Check quadratic equality constraints for neg_max_general penalty
    and gradients.
    """
    N_iter = 50
    dim, M, As, bs, Cs, ds, Fs, gradFs, Gs, gradGs, eps = \
            quadratic_equality(N_iter)
    def f(X):
        return neg_max_penalty(X, M, As, bs, Cs, ds, Fs, Gs)
    def gradf(X):
        return neg_max_grad_penalty(X, M,
                As, bs, Cs, ds, Fs, gradFs, Gs, gradGs, eps)
    X, fX, SUCCEED = run_experiment(f, gradf, dim, N_iter)
    #import pdb
    #pdb.set_trace()


def stress_inequalities(dim, N_iter):
    """
    Stress test the bounded trace solver for
    inequalities.

    With As and bs as below, we specify the probelm

    max penalty(X)
    subject to
        x_ii <= 1/2n
        Tr(X) = x_11 + x_22 + ... + x_nn == 1

    The optimal solution should equal a diagonal matrix with small entries
    for the first n-1 diagonal elements, but a large element (about 1/2)
    for the last element.
    """
    m = dim - 1
    n = 0
    eps = 1./N_iter
    M = compute_scale(m, n, eps)
    As = []
    for i in range(dim-1):
        Ai = np.zeros((dim,dim))
        Ai[i,i] = 1
        As.append(Ai)
    bs = []
    for i in range(dim-1):
        bi = 1./(2*dim)
        bs.append(bi)
    Cs = []
    ds = []
    Fs = []
    gradFs = []
    Gs = []
    gradGs = []
    return m, n, M, As, bs, Cs, ds, eps, Fs, gradFs, Gs, gradGs

def test6a():
    """
    Stress test inequality constraints for log_sum_exp penalty.
    """
    dims = [4,16]
    N_iter = 50
    for dim in dims:
        m, n, M, As, bs, Cs, ds, eps, Fs, gradFs, Gs, gradGs = \
                stress_inequalities(dim, N_iter)
        def f(X):
            return log_sum_exp_penalty(X, M, As, bs, Cs, ds, Fs, Gs)
        def gradf(X):
            return log_sum_exp_grad_penalty(X, M,
                    As, bs, Cs, ds, Fs, gradFs, Gs, gradGs, eps)
        run_experiment(f, gradf, dim, N_iter)

def test6b():
    """
    Stress test inequality constraints for neg_max_sum penatly
    and log_sum_exp gradient.
    """
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
    Stress test inequality constraints for neg_max_penalty
    """
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

def stress_equalities(dim, N_iter):
    """
    Stress test the bounded trace solver for equalities.

    With As and bs as below, we solve the problem

    max penalty(X)
    subject to
        x_ii == 0, i < n
        Tr(X) = x_11 + x_22 + ... + x_nn == 1

    The optimal solution should equal a diagonal matrix with zero entries
    for the first n-1 diagonal elements, but a 1 for the diagonal element.
    """
    m = 0
    n = dim - 1
    eps = 1./N_iter
    M = compute_scale(m, n, eps)
    As = []
    bs = []
    Cs = []
    for j in range(dim-1):
        Cj = np.zeros((dim,dim))
        Cj[j,j] = 1
        Cs.append(Cj)
    ds = []
    for j in range(dim-1):
        dj = 0.
        ds.append(dj)
    Fs = []
    gradFs = []
    Gs = []
    gradGs = []
    return m, n, M, As, bs, Cs, ds, eps, Fs, gradFs, Gs, gradGs

def test7a():
    """
    Stress test equality constraints for log_sum_exp_penalty
    """
    dims = [4,16]
    N_iter = 50
    for dim in dims:
        m, n, M, As, bs, Cs, ds, eps, Fs, gradFs, Gs, gradGs = \
                stress_equalities(dim, N_iter)
        def f(X):
            return log_sum_exp_penalty(X, M, As, bs, Cs, ds, Fs, Gs)
        def gradf(X):
            return log_sum_exp_grad_penalty(X, M,
                        As, bs, Cs, ds, Fs, gradFs, Gs, gradGs, eps)
        run_experiment(f, gradf, dim, N_iter)

def test7b():
    """
    Stress test equality constraints for neg_max_penalty
    """
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
    Stress test equality constraints for neg_max_penalty
    """
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
        #import pdb
        #pdb.set_trace()


def run_experiment(f, gradf, dim, N_iter, alphas=None,DEBUG=False):
    fudge_factor = 5.0
    eps = 1./N_iter
    B = BoundedTraceSolver()
    start = time.clock()
    X = B.solve(f, gradf, dim, N_iter, DEBUG=DEBUG, alphas=alphas)
    elapsed = (time.clock() - start)
    fX = f(X)
    print "\tX:\n", X
    print "\tf(X) = %f" % fX
    SUCCEED = not (fX < -fudge_factor * eps)
    print "\tSUCCEED: " + str(SUCCEED)
    print "\tComputation Time (s): ", elapsed
    return X, fX, SUCCEED
