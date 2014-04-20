from hazan import *
from hazan_penalties import *
import pdb
import time

def test1():
    """
    Do a simple test of the Bounded Trace Solver on function
    f(x)  = -\sum_k x_kk^2 defined above.

    Now do a dummy optimization problem. The
    problem we consider is

        max - sum_k x_k^2
        subject to
            sum_k x_k = 1

    The optimal solution is -1/n, where
    n is the dimension.
    """
    N_iter = 50
    # dimension of square matrix X
    dims = [1,4,16,64]
    for dim in dims:
        print("dim = %d" % dim)
        # Note that H(-f) = 2 I (H is the hessian of f)
        Cf = 2.
        b = BoundedTraceSDPHazanSolver()
        X = b.solve(neg_sum_squares, grad_neg_sum_squares,
                dim, N_iter, Cf=Cf)
        fX = neg_sum_squares(X)
        print("\tTr(X) = %f" % np.trace(X))
        print("\tf(X) = %f" % fX)
        print("\tf* = %f" % (-1./dim))
        print("\t|f(X) - f*| = %f" % (np.abs(fX - (-1./dim))))
        print("\tError Tolerance 1/%d = %f" % (N_iter, 1./N_iter))
        assert np.abs(fX - (-1./dim)) < 1./N_iter
        print("\tError Tolerance Acceptable")

def simple_equality_constraint_test(N_iter, penalty, grad_penalty):
    """
    Check that the bounded trace implementation can handle low dimensional
    equality type constraints for the given penalty function.

    With As and bs as below, we solve the problem

        max penalty(X)
        subject to
          x_11 + 2 x_22 == 1.5
          Tr(X) = x_11 + x_22 == 1.

    We should find penalty(X) >= -eps, and that the above constraints have
    a solution
    """
    m = 0
    n = 1
    dim = 2
    eps = 1./N_iter
    M = compute_scale(m, n, eps)
    def f(X):
        return penalty(X, m, n, M, As, bs, Cs, ds, dim)
    def gradf(X):
        return grad_penalty(X, m, n, M, As, bs, Cs, ds, dim, eps)
    As = []
    bs = []
    Cs = [np.array([[ 1.,  0.],
                    [ 0.,  2.]])]
    ds = [1.5]
    import pdb
    pdb.set_trace()
    run_experiment(f, gradf, dim, N_iter)

def test2():
    """
    Check equality constraints for log_sum_exp constraints
    """
    N_iter = 50
    simple_equality_constraint_test(N_iter, log_sum_exp_penalty,
            log_sum_exp_grad_penalty)

def test3():
    """
    Check equality constraints for neg_max constraints
    """
    N_iter = 50
    simple_equality_constraint_test(N_iter, neg_max_penalty,
            neg_max_grad_penalty)

def simple_constraint_test(N_iter, penalty, grad_penalty):
    """
    Check that the bounded trace implementation can handle low-dimensional
    equality and inequality type constraints for given penalty.

    With As and bs as below, we solve the problem

        max penalty(X)
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
    def f(X):
        return penalty(X, m, n, M, As, bs, Cs, ds, dim)
    def gradf(X):
        return grad_penalty(X, m, n, M, As, bs, Cs, ds, dim, eps)
    As = [np.array([[ 1., 0., 0.],
                    [ 0., 2., 0.],
                    [ 0., 0., 0.]])]
    bs = [1.]
    Cs = [np.array([[ 1.,  0., 0.],
                    [ 0.,  2., 0.],
                    [ 0.,  0., 2.]])]
    ds = [5./3]
    run_experiment(f, gradf, dim, N_iter)

def test4():
    """
    Check equality and inequality constraints for log_sum_exp penalty
    """
    N_iter = 50
    simple_constraint_test(N_iter, log_sum_exp_penalty,
            log_sum_exp_grad_penalty)

def test5():
    """
    Check equality and inequality constraints for neg_max penalty
    """
    N_iter = 50
    simple_constraint_test(N_iter, neg_max_penalty,
            neg_max_grad_penalty)

def stress_test_inequalities(dims, N_iter, penalty, grad_penalty):
    """
    Stress test the bounded trace solver for
    inequalities.

    With As and bs as below, we solve the probelm

    max penalty(X)
    subject to
        x_ii <= 1/2n
        Tr(X) = x_11 + x_22 + ... + x_nn == 1

    The optimal solution should equal a diagonal matrix with small entries
    for the first n-1 diagonal elements, but a large element (about 1/2)
    for the last element.
    """
    for dim in dims:
        m = dim - 1
        n = 0
        eps = 1./N_iter
        M = compute_scale(m, n, eps)
        def f(X):
            return penalty(X, m, n, M, As, bs, Cs, ds, dim)
        def gradf(X):
            return grad_penalty(X, m, n, M, As, bs, Cs, ds, dim, eps)
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
        run_experiment(f, gradf, dim, N_iter)

def test6():
    """
    Stress test inequality constraints for neg_max_penalty
    """
    dims = [4,16]
    N_iter = 50
    stress_test_inequalities(dims, N_iter, neg_max_penalty,
            neg_max_grad_penalty)

def test65():
    """
    Stress test inequality constraints for log_sum_exp penalty.
    """
    pass

def stress_test_equalities(dims, N_iter, penalty, grad_penalty):
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
    for dim in dims:
        m = 0
        n = dim - 1
        eps = 1./N_iter
        M = compute_scale(m, n, eps)
        def f(X):
            return neg_max_penalty(X, m, n, M, As, bs, Cs, ds, dim)
        def gradf(X):
            return neg_max_grad_penalty(X, m, n, M,
                        As, bs, Cs, ds, dim,eps)
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
        run_experiment(f, gradf, dim, N_iter)

def test7():
    """
    TODO: Stress test log_sum_exp once made numerically stable
    Stress test equality constraints for neg_max_penalty
    """
    dims = [4,16]
    N_iter = 50
    stress_test_equalities(dims, N_iter, neg_max_penalty,
            neg_max_grad_penalty)


def stress_test_inequalies_and_equalities(dims, N_iter,
        penalty, grad_penalty):
    """
    Stress test the bounded trace solver for both equalities and
    inequalities.

    With As and bs as below, we solve the problem

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
    for dim in dims:
        m = dim - 2
        n = dim**2 - dim
        As = []
        M = compute_scale(m, n, eps)
        def f(X):
            return penalty(X, m, n, M, As, bs, Cs, ds, dim)
        def gradf(X):
            return grad_penalty(X, m, n, M,
                        As, bs, Cs, ds, dim,eps)
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
        run_experiment(f, gradf, dim, N_iter)

def test8():
    """
    TODO: Stress test log_sum_exp once made numerically stable
    Stress test equality constraints for neg_max_penalty
    """
    dims = [4, 16]
    N_iter = 50
    stress_test_inequalies_and_equalities(dims, N_iter,
            neg_max_penalty, neg_max_grad_penalty)

def run_experiment(f, gradf, dim, N_iter):
    fudge_factor = 5.0
    eps = 1./N_iter
    B = BoundedTraceSDPHazanSolver()
    start = time.clock()
    X = B.solve(f, gradf, dim, N_iter, DEBUG=False)
    elapsed = (time.clock() - start)
    fX = f(X)
    print "\tX:\n", X
    print "\tf(X) = %f" % fX
    FAIL = (fX < -fudge_factor * eps)
    print "\tSUCCEED: " + str(not FAIL)
    print "\tComputation Time (s): ", elapsed

if __name__ == "__main__":
    # TODO: change these tests to Nosetests style
    #test1()
    #test2()
    test3()
    #test4()
    #test5()
    #test6()
    #test7()
    #test8()
    pass
