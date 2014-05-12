import numpy as np
from constraints import *

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

# - log det R + Tr(RB)
def log_det_tr(X, B):
    """
    minimize -log det R + Tr(RB)
          --------------
         |D-ADA.T  I    |
    X =  |   I     R    |
         |            R |
          --------------
    X is PSD
    """
    (dim, _) = np.shape(X)
    block_dim = int(dim/3)
    (D_ADA_T_cds, I_1_cds, I_2_cds, R_1_cds, R_2_cds) = \
            Q_coords(block_dim)
    R1 = get_entries(X, R_1_cds)
    R2 = get_entries(X, R_2_cds)
    try:
        val1 = -np.log(np.linalg.det(R1)) + np.trace(np.dot(R1, B))
        val2 = -np.log(np.linalg.det(R2)) + np.trace(np.dot(R2, B))
        val = val1 + val2
    except FloatingPointError:
        return -np.inf
    return val

# grad - log det R = -R^{-1} = -Q (see Boyd and Vandenberge, A4.1)
# grad tr(RB) = B^T
def grad_log_det_tr(X, B):
    """
    minimize -log det R + Tr(RB)
          --------------
         |D-ADA.T  I    |
    X =  |   I     R    |
         |            R |
          --------------
    X is PSD
    """
    (dim, _) = np.shape(X)
    block_dim = int(dim/3)
    (D_ADA_T_cds, I_1_cds, I_2_cds, R_1_cds, R_2_cds) = \
            Q_coords(block_dim)
    grad = np.zeros(np.shape(X))
    R1 = get_entries(X, R_1_cds)
    R2 = get_entries(X, R_2_cds)
    Q1 = np.linalg.inv(R1)
    Q2 = np.linalg.inv(R2)
    gradR1 = -Q1.T + B.T
    gradR2 = -Q2.T + B.T
    set_entries(grad, R_1_cds, gradR1)
    set_entries(grad, R_2_cds, gradR2)
    return grad
