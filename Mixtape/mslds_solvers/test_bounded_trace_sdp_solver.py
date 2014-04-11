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

def test1():
    """
     Do a simple test of the Bounded Trace Solver on function
     f(x)  = -\sum_k x_kk^2 defined above.
    """
    # dimension of square matrix X
    dims = [1,2,4,8,16,32,64,128]
    for dim in dims:
        print("dim = %d" % dim)
        # Note that H(-f) = 2 I (H is the hessian of f)
        Cf = 2.
        N_iter = 2 * dim
        # Now do a dummy optimization problem. The
        # problem we consider is
        # max - \sum_k x_k^2
        # such that \sum_k x_k = 1
        # The optimal solution is -1/n, where
        # n is the dimension.
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
    """
    m = 0
    n = 1
    dim = 2
    N_iter = 50
    eps = 1./N_iter
    if m > 0:
        M = np.max((np.log(m), 1.))/eps
        print("M", M)
    if n > 0:
        N = np.max((np.log(n), 1.))/(eps**2)
        print("N", N)
    def f(X):
        """
        f(X) = -(1/M) log(sum_{i=1}^m exp(M*(Tr(Ai,X) - bi))) +
               -(1/N) log( sum_{j=1}^n exp(-M*(Tr(Cj,X) - dj)^ 2))
        """
        s = 0.
        r = 0.
        retval = 0.
        for i in range(m):
            Ai = As[i]
            bi = bs[i]
            s += np.exp(M*(np.trace(np.dot(Ai,X)) - bi))
        if m > 0:
            retval += -(1.0/M) * np.log(s)
        for j in range(n):
            Cj = Cs[j]
            dj = ds[j]
            #r += np.exp(-M*(np.trace(np.dot(Cj,X) - dj)**2))
            #print("Cj\n", Cj)
            #print("X\n", X)
            #print("(np.trace(np.dot(Cj,X) - dj)**2): ",
            #        (np.trace(np.dot(Cj,X)) - dj)**2)
            r += np.exp(N*(np.trace(np.dot(Cj,X)) - dj)**2)
        if n > 0:
            #retval += -(1.0/N) * np.log(r)
            retval += -(1.0/N) * np.log(r)
        return retval
    def gradf(X):
        """
        Computes grad f(X) = -(1/M) * f' / f where
          f' = (sum_{i=1}^m exp(M*(Tr(Ai, X) - bi)) * (M * Ai.T)
    + sum_{j=1}^n exp(-M(Tr(Cj,X) - dj)**2) * (-2M(Tr(Cj,X) - dj)) * Cj.T
          f  = (sum_{i=1}^m exp(M*(Tr(Ai,X) - bi))
            + sum_{i=1}^n exp(-M(Tr(Cj,X) - dj)**2))
        """
        retval = 0.
        num1 = 0.
        denom1 = 0.
        for i in range(m):
            Ai = As[i]
            bi = bs[i]
            if dim >= 2:
                num1 += np.exp(M*(np.trace(np.dot(Ai,X)) - bi))*(M*Ai.T)
                denom1 += np.exp(M*(np.trace(np.dot(Ai,X)) - bi))
            else:
                num1 += np.exp(M*(Ai*X - bi))*(M*Ai.T)
                denom1 += np.exp(M*(Ai*X - bi))
        if m > 0:
            retval += (-1.0/M) * num1/denom1
        num2 = 0.
        denom2 = 0.
        for j in range(n):
            Cj = Cs[j]
            dj = ds[j]
            if dim >= 2:
                #num2 += (np.exp(-N*(np.trace(np.dot(Cj,X)) - dj)**2)*
                #        (-2*N*(np.trace(np.dot(Cj,X)) - dj))*
                #        Cj.T)
                #denom2 += np.exp(-N*(np.trace(np.dot(Cj,X)) - dj)**2)
                num2 += (np.exp(N*(np.trace(np.dot(Cj,X)) - dj)**2)*
                        (2*N*(np.trace(np.dot(Cj,X)) - dj))*
                        Cj.T)
                denom2 += np.exp(N*(np.trace(np.dot(Cj,X)) - dj)**2)
            else:
                #num2 += np.exp(-N*(cj*x - dj)**2)*(-2*m*(cj*x - dj))*cj.t
                #denom2 += np.exp(-N*(cj*x - dj)**2)
                num2 += np.exp(N*(Cj*x - dj)**2)*(2*N*(Cj*x - dj))*Cj.T
                denom2 += np.exp(N*(Cj*x - dj)**2)
        if n > 0:
            #retval += (-1.0/N) * num2/denom2
            retval += -(1.0/N) * num2/denom2
        return retval
    # With As and bs as below, we specify
    # x_11 + 2 x_22 <= 1.5
    # -x_11 - 2 x_22 <= 1.5
    # That is, we encode equality x_11 + 2x_22 = 1.5 as two
    #As = [np.array([[ 1.,  0.],
    #                [ 0.,  2.]]),
    #      np.array([[-1., -0.],
    #                [-0., -2.]])]
    #bs = [1.5, 1.5]
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
    #pdb.set_trace()
    pass

if __name__ == "__main__":
    #test1()
    test2()
