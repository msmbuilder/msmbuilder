from hazan import *

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
        print("\tf(X) = %f" % fX)
        print("\tf* = %f" % (-1./dim))
        print("\t|f(X) - f*| = %f" % (np.abs(fX - (-1./dim))))
        print("\tError Tolerance 1/%d = %f" % (N_iter, 1./N_iter))
        assert np.abs(fX - (-1./dim)) < 1./N_iter
        print("\tError Tolerance Acceptable")

if __name__ == "__main__":
    test1()
