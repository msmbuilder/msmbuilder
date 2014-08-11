from __future__ import division
from __future__ import absolute_import
import numpy as np
from .constraints import *

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

def trace_obj(X):
    """
    Computes tr(X)
    """
    return np.trace(X)

def grad_trace_obj(X):
    """
    Computes grad tr(X) = I
    """
    (dim, _) = np.shape(X)
    return np.eye(dim)

# - log det R + Tr(RB)
def log_det_tr(X, B):
    """
    minimize -log det R + Tr(RB)
          -------------------
         |D-ADA.T  I         |
    X =  |   I     R         |
         |            D   cI |
         |           cI   R  |
          -------------------
    X is PSD
    """
    with np.errstate(divide='raise'):
        (dim, _) = np.shape(X)
        block_dim = int(dim/4)
        (D_ADA_T_cds, I_1_cds, I_2_cds, R_1_cds, 
            D_cds, c_I_1_cds, c_I_2_cds, R_2_cds) = \
                Q_coords(block_dim)
        R1 = get_entries(X, R_1_cds)
        R2 = get_entries(X, R_2_cds)
        # Need to avoid ill-conditioning of R1, R2
        R1 = R1 + (1e-5) * np.eye(block_dim)
        R2 = R2 + (1e-5) * np.eye(block_dim)
        try:
            #val1 = -np.log(np.linalg.det(R1)) + np.trace(np.dot(R1, B))
            L1 = np.linalg.cholesky(R1)
            log_det1 = 2*np.sum(np.log(np.diag(L1)))
            val1 = -log_det1 + np.trace(np.dot(R1, B))
        except FloatingPointError:
            if ((np.linalg.det(R1) < np.finfo(np.float).eps)
                or not np.isfinite(np.linalg.det(R1))):
                val1 = np.inf
        try:
            #val2 = -np.log(np.linalg.det(R2)) + np.trace(np.dot(R2, B))
            L2 = np.linalg.cholesky(R2)
            log_det2 = 2*np.sum(np.log(np.diag(L2)))
            val2 = -log_det2 + np.trace(np.dot(R2, B))
        except FloatingPointError:
            if ((np.linalg.det(R2) < np.finfo(np.float).eps)
                or not np.isfinite(np.linalg.det(R2))):
                val2 = np.inf
        val = val1 + val2 
        return val 

# grad - log det R = -R^{-1} = -Q (see Boyd and Vandenberge, A4.1)
# grad tr(RB) = B^T
def grad_log_det_tr(X, B):
    """
    minimize -log det R + Tr(RB)
          -------------------
         |D-ADA.T  I         |
    X =  |   I     R         |
         |            D   cI |
         |           cI   R  |
          -------------------
    X is PSD
    """
    (dim, _) = np.shape(X)
    block_dim = int(dim/4)
    (D_ADA_T_cds, I_1_cds, I_2_cds, R_1_cds, 
        D_cds, c_I_1_cds, c_I_2_cds, R_2_cds) = \
            Q_coords(block_dim)
    grad = np.zeros(np.shape(X))
    R1 = get_entries(X, R_1_cds)
    R2 = get_entries(X, R_2_cds)
    # Need to avoid ill-conditioning of R1, R2
    R1 = R1 + (1e-5) * np.eye(block_dim)
    R2 = R2 + (1e-5) * np.eye(block_dim)
    Q1 = np.linalg.inv(R1)
    Q2 = np.linalg.inv(R2)
    gradR1 = -Q1.T + B.T
    gradR2 = -Q2.T + B.T
    set_entries(grad, R_1_cds, gradR1)
    set_entries(grad, R_2_cds, gradR2)
    return grad

# Tr [ Q^{-1} ([C - B] A.T + A [C - B].T + A E A.T]
def A_dynamics(X, dim, C, B, E, Qinv):
    """
    min Tr [ Q^{-1} ([C - B] A.T + A [C - B].T + A E A.T]

          --------------------
         | D-Q    A           |
    X =  | A.T  D^{-1}        |
         |              I   A |
         |             A.T  I |
          --------------------
    X is PSD
    """
    (D_Q_cds, Dinv_cds, I_1_cds, I_2_cds,
        A_1_cds, A_T_1_cds, A_2_cds, A_T_2_cds) = A_coords(dim)

    A_1 = get_entries(X, A_1_cds)
    A_T_1 = get_entries(X, A_T_1_cds)
    A_2 = get_entries(X, A_2_cds)
    A_T_2 = get_entries(X, A_T_2_cds)
    def obj(A):
        return np.dot(Qinv, (np.dot(C-B, A.T) + np.dot(C-B, A.T).T
                            + np.dot(A, np.dot(E, A.T))))
    term_1, term_T_1, term_2, term_T_2 = \
            obj(A_1), obj(A_T_1.T), obj(A_2), obj(A_T_2)
    return np.trace(term_1+term_T_1+term_2+term_T_2)

# grad Tr [Q^{-1} (C - B) A.T] = Q^{-1} (C - B)
# grad Tr [Q^{-1} A [C - B].T] = Q^{-T} (C - B)
# grad Tr [Q^{-1} A E A.T] = Q^{-T} A E.T + Q^{-1} A E
def grad_A_dynamics(X, dim, C, B, E, Qinv):
    """
    min Tr [ Q^{-1} ([C - B] A.T + A [C - B].T + A E A.T]

          --------------------
         | D-Q    A           |
    X =  | A.T  D^{-1}        |
         |              I   A |
         |             A.T  I |
          --------------------
    X is PSD
    """
    (D_Q_cds, Dinv_cds, I_1_cds, I_2_cds,
        A_1_cds, A_T_1_cds, A_2_cds, A_T_2_cds) = A_coords(dim)
    grad = np.zeros(np.shape(X))
    A_1 = get_entries(X, A_1_cds)
    A_T_1 = get_entries(X, A_T_1_cds)
    A_2 = get_entries(X, A_2_cds)
    A_T_2 = get_entries(X, A_T_2_cds)
    def grad_obj(A):
        grad_term1 = np.dot(Qinv, C-B)
        grad_term2 = np.dot(Qinv.T, C-B)
        grad_term3 = np.dot(Qinv.T, np.dot(A, E.T)) + \
                        np.dot(Qinv, np.dot(A, E))
        gradA = grad_term1 + grad_term2 + grad_term3
        return gradA
    gradA_1, gradA_T_1, gradA_2, gradA_T_2 = \
            (grad_obj(A_1), grad_obj(A_T_1.T),
                grad_obj(A_2), grad_obj(A_T_2.T))
    set_entries(grad, A_1_cds, gradA_1)
    set_entries(grad, A_T_1_cds, gradA_T_1.T)
    set_entries(grad, A_2_cds, gradA_2)
    set_entries(grad, A_T_2_cds, gradA_T_2.T)
    return grad
