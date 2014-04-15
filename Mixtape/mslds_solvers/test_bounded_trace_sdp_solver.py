from hazan import *
import pdb

def f(x):
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

def gradf(x):
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
        s += np.exp(M*(np.trace(np.dot(Ai,X)) - bi))
    for j in range(n):
        Cj = Cs[j]
        dj = ds[j]
        r += np.exp(M*(np.trace(np.dot(Cj,X)) - dj)**2)
    if m + n > 0:
        retval += -(1.0/M) * np.log(s + r)
    return retval

def stable_penalty(X, m, n, M, As, bs, Cs, ds):
    """
    Note that penalty(X) roughly equals

     max_{i,j} {{exp(M*(Tr(Ai,X) - bi))}, {exp(M*(Tr(Cj,X) - dj)^2)}}

    This function computes and returns this quantity.
    """
    pass


def stable_grad_penalty(X, m, n, M, As, bs, Cs, ds, dim):
    """
    Note that penalty(X) roughly equals

     max_{i,j} {{exp(M*(Tr(Ai,X) - bi))}, {exp(M*(Tr(Cj,X) - dj)^2)}}

    Thus, we can approximate the gradient by picking the term such that
    violation exp(M*(Tr(Ai,X) - bi)) or exp(M*(Tr(Cj,X) - dj)^2) is
    largest and then returning the gradient of that terms alone
    """
    retval = 0.
    log_nums = np.zeros(n+m)
    log_denoms = np.zeros(n+m)
    ind = 0
    for i in range(m):
        Ai = As[i]
        bi = bs[i]
        if dim >= 2:
            log_nums[count] += (M*(np.trace(np.dot(Ai,X)) - bi)
                                + np.log(M*Ai.T))
            log_denoms[count] += M*(np.trace(np.dot(Ai,X)) - bi)
        else:
            log_nums[count] += M*(Ai*X - bi) + np.log(M*Ai.T)
            log_denoms[count] += np.exp(M*(Ai*X - bi))
        count += ind
    for j in range(n):
        Cj = Cs[j]
        dj = ds[j]
        if dim >= 2:
            log_nums[count] += (np.exp(M*(np.trace(np.dot(Cj,X)) - dj)**2)*
                    (2*M*(np.trace(np.dot(Cj,X)) - dj))*
                    Cj.T)
            log_denoms[count] += np.exp(M*(np.trace(np.dot(Cj,X)) - dj)**2)
        else:
            log_nums[count] += np.exp(M*(Cj*x - dj)**2)*(2*M*(Cj*x - dj))*Cj.T
            log_denoms[count] += np.exp(M*(Cj*x - dj)**2)
        count += ind
    if m + n > 0:
        retval += -(1.0/M) * num/denom
    import pdb
    pdb.set_trace()
    return retval

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
        X = b.solve(f, gradf, dim, N_iter, Cf=Cf)
        #print("\tX:")
        #print X
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
    Check that the bounded trace implementation can handle equality and
    inequality type constraings

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

if __name__ == "__main__":
    #test1()
    #test2()
    #test3()
    m = 1
    n = 1
    X = np.eye(3)
    As = [np.array([[ 1., 0., 0.],
                    [ 0., 2., 0.],
                    [ 0., 0., 0.]])]
    bs = [1.]
    Cs = [np.array([[ 1.,  0., 0.],
                    [ 0.,  2., 0.],
                    [ 0.,  0., 2.]])]
    ds = [5./3]
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
    val = grad_penalty(X, m, n, M, As, bs, Cs, ds, dim)
