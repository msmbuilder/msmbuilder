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
    # Do a simple test of the Bounded Trace Solver
    dim = 8
    # Note that H(-f) = 2 I (H is the hessian)
    Cf = 2.
    N_iter = 100
    # Now do a dummy optimization problem. The
    # problem we consider is
    # max - \sum_k x_k^2
    # such that \sum_k x_k = 1
    # The optimal solution is -1/n, where
    # n is the dimension.
    b = BoundedTraceSDPHazanSolver()
    X = b.solve(f, gradf, dim, N_iter, Cf=Cf)
    print("X:")
    print X
    fX = f(X)
    print("f(X) = %f" % fX)
    assert np.abs(fX - (-1./dim)) < 1./N_iter

if __name__ == "__main__":
    test1()
