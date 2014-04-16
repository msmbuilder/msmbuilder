from hazan import *
import pdb
import time

def neg_sum_squares(x):
    """
    Computes f(x) = -\sum_k x_kk^2

    Parameters
    __________
    x: numpy.ndarray
    """
    (N, _) = np.shape(x)
    retval = 0.
    for i in range(N):
        retval += -x[i,i]**2
    return retval

def grad_neg_sum_squares(x):
    """
    Computes grad(-\sum_k x_kk^2)

    Parameters
    __________
    x: numpy.ndarray
    """
    (N, _) = np.shape(x)
    G = np.zeros((N,N))
    for i in range(N):
        G[i,i] += -2.*x[i,i]
    return G

def penalty(X, m, n, M, As, bs, Cs, ds):
    """
    Computes
    f(X) = -(1/M) log(sum_{i=1}^m exp(M*(Tr(Ai,X) - bi))
                    + sum_{j=1}^n exp(M*(Tr(Cj,X) - dj)^2))

    Parameters
    __________

    m: int
        Number of inequaltity constraints
    n: int
        Number of equality constraints
    M: float
        Rescaling Factor
    """
    s = 0.
    r = 0.
    retval = 0.
    for i in range(m):
        Ai = As[i]
        bi = bs[i]
        if dim >= 2:
            s += np.exp(M*(np.trace(np.dot(Ai,X)) - bi))
        else:
            s += np.exp(M*(Ai*X - bi))
    for j in range(n):
        Cj = Cs[j]
        dj = ds[j]
        if dim >= 2:
            r += np.exp(M*(np.trace(np.dot(Cj,X)) - dj)**2)
        else:
            r += np.exp(M*(Cj*X - dj)**2)
    if m + n > 0:
        retval += -(1.0/M) * np.log(s + r)
    return retval

def neg_max_penalty(X, m, n, M, As, bs, Cs, ds, dim):
    """
    Note that penalty(X) roughly equals

     -max(max_i {Tr(Ai,X) - bi}, max_j{|Tr(Cj,X) - dj|})

    This function computes and returns this quantity.
    """
    penalties = np.zeros(n+m)
    count = 0
    for i in range(m):
        Ai = As[i]
        bi = bs[i]
        if dim >= 2:
            penalties[i] = np.trace(np.dot(Ai,X)) - bi
        else:
            penalties[i] = Ai*X - bi
    for j in range(n):
        Cj = Cs[j]
        dj = ds[j]
        if dim >= 2:
            penalties[j+m] += np.abs(np.trace(np.dot(Cj,X)) - dj)
        else:
            penalties[j+m] += np.abs(Cj*x - dj)
    return -np.amax(penalties)

def neg_max_grad_penalty(X, m, n, M, As, bs, Cs, ds, dim, eps):
    """
    Note that penalty(X) roughly equals

     max(max_i {Tr(Ai,X) - bi}, max_j{(Tr(Cj,X) - dj)^2})

    The subgradient of this quantity is given by

    Conv{{A_i | penalty(X) = Tr(Ai,X) - bi} union
         { sign(Tr(Cj,X) - dj) 2 Cj  | penalty(X) = |Tr(Cj,X) - dj| }}

    We use a weak subdifferential calculus that averages the gradients
    of all violated constraints.
    """
    penalties = np.zeros(n+m)
    count = 0
    for i in range(m):
        Ai = As[i]
        bi = bs[i]
        if dim >= 2:
            penalties[i] = np.trace(np.dot(Ai,X)) - bi
        else:
            penalties[i] = Ai*X - bi
    for j in range(n):
        Cj = Cs[j]
        dj = ds[j]
        if dim >= 2:
            penalties[j+m] += np.abs(np.trace(np.dot(Cj,X)) - dj)
        else:
            penalties[j+m] += np.abs(Cj*x - dj)
    #ind = np.argmax(penalties)
    inds = [ind for ind in range(n+m) if penalties[ind] > eps]

    grad = np.zeros(np.shape(X))
    for ind in inds:
        if ind < m:
            Ai =  As[ind]
            grad += Ai
        else:
            Cj = Cs[ind - m]
            val = np.trace(np.dot(Cj,X)) - dj
            if val < 0:
                grad += - Cj
            elif val > 0:
                grad += Cj
    # Average by num entries
    grad = grad / max(len(inds), 1.)
    # Take the negative since our function is -max{..}
    return -grad

def grad_penalty(X, m, n, M, As, bs, Cs, ds, dim):
    """
    Computes grad f(X) = -(1/M) * c' / c where
      c' = (sum_{i=1}^m exp(M*(Tr(Ai, X) - bi)) * (M * Ai.T)
            + sum_{j=1}^n exp(M(Tr(Cj,X) - dj)**2)
                            * (2M(Tr(Cj,X) - dj)) * Cj.T)
      c  = (sum_{i=1}^m exp(M*(Tr(Ai,X) - bi))
            + sum_{i=1}^n exp(M(Tr(Cj,X) - dj)**2))

    Need Ai and Cj to be symmetric real matrices
    """
    retval = 0.
    nums = 0.
    denoms = 0.
    for i in range(m):
        Ai = As[i]
        bi = bs[i]
        if dim >= 2:
            num += np.exp(M*(np.trace(np.dot(Ai,X)) - bi))*(M*Ai.T)
            denom += np.exp(M*(np.trace(np.dot(Ai,X)) - bi))
        else:
            num += np.exp(M*(Ai*X - bi))*(M*Ai.T)
            denom += np.exp(M*(Ai*X - bi))
    for j in range(n):
        Cj = Cs[j]
        dj = ds[j]
        if dim >= 2:
            num += (np.exp(M*(np.trace(np.dot(Cj,X)) - dj)**2)*
                    (2*M*(np.trace(np.dot(Cj,X)) - dj))*
                    Cj.T)
            denom += np.exp(M*(np.trace(np.dot(Cj,X)) - dj)**2)
        else:
            num += np.exp(M*(Cj*x - dj)**2)*(2*M*(Cj*x - dj))*Cj.T
            denom += np.exp(M*(Cj*x - dj)**2)
    if m + n > 0:
        retval += -(1.0/M) * num/denom
    import pdb
    pdb.set_trace()
    return retval

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
    # dimension of square matrix X
    dims = [1,2,4,8,16,32,64,128]
    for dim in dims:
        print("dim = %d" % dim)
        # Note that H(-f) = 2 I (H is the hessian of f)
        Cf = 2.
        N_iter = 2 * dim
        b = BoundedTraceSDPHazanSolver()
        X = b.solve(neg_sum_squares, grad_neg_sum_squares,
                dim, N_iter, Cf=Cf)
        fX = f(X)
        print("\tTr(X) = %f" % np.trace(X))
        print("\tf(X) = %f" % fX)
        print("\tf* = %f" % (-1./dim))
        print("\t|f(X) - f*| = %f" % (np.abs(fX - (-1./dim))))
        print("\tError Tolerance 1/%d = %f" % (N_iter, 1./N_iter))
        assert np.abs(fX - (-1./dim)) < 1./N_iter
        print("\tError Tolerance Acceptable")


def test2():
    """
    Check that the bounded trace implementation can handle equality type
    constraints.

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
    N_iter = 50
    eps = 1./N_iter
    M = 1.
    if m + n > 0:
        M = 0.
        if m > 0:
            M += np.max((np.log(m), 1.))/eps
        if n > 0:
            M += np.max((np.log(n), 1.))/(eps**2)
        print("M", M)
    def f(X):
        return penalty(X, m, n, M, As, bs, Cs, ds)
    def gradf(X):
        return grad_penalty(X, m, n, M, As, bs, Cs, ds, dim)
    As = []
    bs = []
    Cs = [np.array([[ 1.,  0.],
                    [ 0.,  2.]])]
    ds = [1.5]
    B = BoundedTraceSDPHazanSolver()
    X = B.solve(f, gradf, dim, N_iter)
    fX = f(X)
    print("X:")
    print X
    print("f(X) = %f" % (fX))
    FAIL = (fX < -eps)
    print("FAIL: " + str(FAIL))

def test3():
    """
    Check that the bounded trace implementation can handle equality type
    constraints with the neg_max penalty

    With As and bs as below, we solve the problem

        max neg_max_penalty(X)
        subject to
          x_11 + 2 x_22 == 1.5
          Tr(X) = x_11 + x_22 == 1.

    We should find neg_max_penalty(X) >= -eps, and that the above
    constraints have a solution
    """
    m = 0
    n = 1
    dim = 2
    N_iter = 50
    eps = 1./N_iter
    M = 1.
    if m + n > 0:
        M = 0.
        if m > 0:
            M += np.max((np.log(m), 1.))/eps
        if n > 0:
            M += np.max((np.log(n), 1.))/(eps**2)
        print("M", M)
    def f(X):
        return neg_max_penalty(X, m, n, M, As, bs, Cs, ds, dim)
    def gradf(X):
        return neg_max_grad_penalty(X, m, n, M, As, bs, Cs, ds, dim, eps)
    As = []
    bs = []
    Cs = [np.array([[ 1.,  0.],
                    [ 0.,  2.]])]
    ds = [1.5]
    B = BoundedTraceSDPHazanSolver()
    X = B.solve(f, gradf, dim, N_iter)
    fX = f(X)
    print("X:")
    print X
    print("f(X) = %f" % (fX))
    FAIL = (fX < -eps)
    print("FAIL: " + str(FAIL))

def test4():
    """
    Check that the bounded trace implementation can handle equality and
    inequality type constraints

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
    N_iter = 50
    eps = 1./N_iter
    M = 1.
    if m + n > 0:
        M = 0.
        if m > 0:
            M += np.max((np.log(m), 1.))/eps
        if n > 0:
            M += np.max((np.log(n), 1.))/(eps**2)
        print("M", M)
    def f(X):
        return penalty(X, m, n, M, As, bs, Cs, ds)
    def gradf(X):
        return grad_penalty(X, m, n, M, As, bs, Cs, ds, dim)
    As = [np.array([[ 1., 0., 0.],
                    [ 0., 2., 0.],
                    [ 0., 0., 0.]])]
    bs = [1.]
    Cs = [np.array([[ 1.,  0., 0.],
                    [ 0.,  2., 0.],
                    [ 0.,  0., 2.]])]
    ds = [5./3]
    B = BoundedTraceSDPHazanSolver()
    X = B.solve(f, gradf, dim, N_iter)

def test5():
    """
    Check that the bounded trace implementation can handle equality and
    inequality type constraints with neg_max_penalty

    With As and bs as below, we solve the problem

        max neg_max_penalty(X)
        subject to
            x_11 + 2 x_22 <= 1
            x_11 + 2 x_22 + 2 x_33 == 5/3
            Tr(X) = x_11 + x_22 + x_33 == 1
    """
    m = 1
    n = 1
    dim = 3
    N_iter = 50
    eps = 1./N_iter
    M = 1.
    if m + n > 0:
        M += np.max((np.log(m), 1.))/eps
        print("M", M)
    def f(X):
        return neg_max_penalty(X, m, n, M, As, bs, Cs, ds, dim)
    def gradf(X):
        return neg_max_grad_penalty(X, m, n, M, As, bs, Cs, ds, dim, eps)
    As = [np.array([[ 1., 0., 0.],
                    [ 0., 2., 0.],
                    [ 0., 0., 0.]])]
    bs = [1.]
    Cs = [np.array([[ 1.,  0., 0.],
                    [ 0.,  2., 0.],
                    [ 0.,  0., 2.]])]
    ds = [5./3]
    B = BoundedTraceSDPHazanSolver()
    X = B.solve(f, gradf, dim, N_iter)
    fX = f(X)
    print("X:")
    print X
    print("f(X) = %f" % (fX))
    FAIL = (fX < -eps)
    print("FAIL: " + str(FAIL))

def test6():
    """
    Stress test the bounded trace solver for
    inequalities.

    With As and bs as below, we solve the probelm

    max neg_max_penalty(X)
    subject to
        x_ii <= 1/2n
        Tr(X) = x_11 + x_22 + ... + x_nn == 1

    The optimal solution should equal a diagonal matrix with small entries
    for the first n-1 diagonal elements, but a large element (about 1/2)
    for the last element.
    """
    dims = [50]
    for dim in dims:
        m = dim - 1
        n = 0
        N_iter = 1000
        eps = 1./N_iter
        M = 1.
        if m + n > 0:
            M = np.max((np.log(m+n), 1.))/eps
            print("M", M)
        def f(X):
            return neg_max_penalty(X, m, n, M, As, bs, Cs, ds, dim)
        def gradf(X):
            return neg_max_grad_penalty(X, m, n, M,
                        As, bs, Cs, ds, dim,eps)
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
        B = BoundedTraceSDPHazanSolver()
        start = time.clock()
        X = B.solve(f, gradf, dim, N_iter, DEBUG=False)
        elapsed = (time.clock() - start)
        fX = f(X)
        print "\tX:\n", X
        print "\tf(X) = %f" % fX
        FAIL = (fX < -eps)
        print "\tFAIL: " + str(FAIL)
        print "\tComputation Time (s): ", elapsed
        #import pdb
        #pdb.set_trace()

def test7():
    """
    Stress test the bounded trace solver for equalities.

    With As and bs as below, we solve the probelm

    max neg_max_penalty(X)
    subject to
        x_ii == 0, i < n
        Tr(X) = x_11 + x_22 + ... + x_nn == 1

    The optimal solution should equal a diagonal matrix with zero entries
    for the first n-1 diagonal elements, but a 1 for the diagonal element.
    """
    dims = [20]
    for dim in dims:
        m = 0
        n = dim - 1
        N_iter = 1000
        eps = 1./N_iter
        M = 1.
        if m + n > 0:
            M = np.max((np.log(m+n), 1.))/eps
            print("M", M)
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
        B = BoundedTraceSDPHazanSolver()
        start = time.clock()
        X = B.solve(f, gradf, dim, N_iter, DEBUG=False)
        elapsed = (time.clock() - start)
        fX = f(X)
        print "\tX:\n", np.around(X, decimals=4)
        print "\tf(X) = %f" % fX
        FAIL = (fX < -eps)
        print "\tFAIL: " + str(FAIL)
        print "\tComputation Time (s): ", elapsed
        #import pdb
        #pdb.set_trace()

def test8():
    """
    Stress test the bounded trace solver for both equalities and
    inequalities.

    With As and bs as below, we solve the probelm

    max neg_max_penalty(X)
    subject to
        x_ij == 0, i != j
        x11
        Tr(X) = x_11 + x_22 + ... + x_nn == 1

    The optimal solution should equal a diagonal matrix with zero entries
    for the first n-1 diagonal elements, but a 1 for the diagonal element.
    """


if __name__ == "__main__":
    #test1()
    #test2()
    #test3()
    #test5()
    #test6()
    test7()
    #m = 1
    #n = 1
    #X = np.eye(3)
    #As = [np.array([[ 1., 0., 0.],
    #                [ 0., 2., 0.],
    #                [ 0., 0., 0.]])]
    #bs = [1.]
    #Cs = [np.array([[ 1.,  0., 0.],
    #                [ 0.,  2., 0.],
    #                [ 0.,  0., 2.]])]
    #ds = [5./3]
    #dim = 3
    #N_iter = 50
    #eps = 1./N_iter
    #M = 1.
    #if m + n > 0:
    #    M = 0.
    #    if m > 0:
    #        M += np.max((np.log(m), 1.))/eps
    #    if n > 0:
    #        M += np.max((np.log(n), 1.))/(eps**2)
    #    print("M", M)
    #val = grad_penalty(X, m, n, M, As, bs, Cs, ds, dim)
