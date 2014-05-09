import numpy as np

def neg_sum_squares(x):
    """
    Computes f(x) = -\sum_k x_kk^2. Useful for debugging.

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
    Computes grad(-\sum_k x_kk^2). Useful for debugging.

    Parameters
    __________
    x: numpy.ndarray
    """
    (N, _) = np.shape(x)
    G = np.zeros((N,N))
    for i in range(N):
        G[i,i] += -2.*x[i,i]
    return G

