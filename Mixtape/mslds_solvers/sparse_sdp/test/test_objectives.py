import sys
sys.path.append("..")
import numpy as np
from utils import numerical_derivative
from objectives import *
from constraints import *

def test_tr():
    dims = [1, 5, 10]
    N_rand = 10
    tol = 1e-3
    eps = 1e-4
    for dim in dims:
        for i in range(N_rand):
            X = np.random.rand(dim, dim)
            val = trace_obj(X)
            grad = grad_trace_obj(X)
            num_grad = numerical_derivative(trace_obj, X, eps)
            assert np.sum(np.abs(grad - num_grad)) < tol

def test_sum_squares():
    dims = [1, 5, 10]
    N_rand = 10
    tol = 1e-3
    eps = 1e-4
    for dim in dims:
        for i in range(N_rand):
            X = np.random.rand(dim, dim)
            val = neg_sum_squares(X)
            grad = grad_neg_sum_squares(X)
            num_grad = numerical_derivative(neg_sum_squares, X, eps)
            assert np.sum(np.abs(grad - num_grad)) < tol

def test_log_det():
    dims = [4]
    N_rand = 10
    tol = 1e-3
    eps = 1e-4
    for dim in dims:
        block_dim = int(dim/4)
        (D_ADA_T_cds, I_1_cds, I_2_cds, R_1_cds, 
            D_cds, c_I_1_cds, c_I_2_cds, R_2_cds) = \
                Q_coords(block_dim)
        # Generate initial data
        B = np.random.rand(block_dim, block_dim)
        #B = np.eye(block_dim)
        def obj(X):
            return log_det_tr(X, B)
        def grad_obj(X):
            return grad_log_det_tr(X, B)
        for i in range(N_rand):
            X = np.random.rand(dim, dim)
            R1 = get_entries(X, R_1_cds)
            R2 = get_entries(X, R_2_cds)
            if (np.linalg.det(R1) <= 0 or np.linalg.det(R2) <= 0):
                "Continue!"
                continue
            val = obj(X)
            grad = grad_obj(X)
            num_grad = numerical_derivative(obj, X, eps)
            diff = np.sum(np.abs(grad - num_grad))
            if diff >= tol:
                print "grad:\n", grad
                print "num_grad:\n", num_grad
                print "diff: ", diff
            assert diff < tol

def test_A_dynamics():
    dims = [4, 8]
    N_rand = 10
    tol = 1e-3
    eps = 1e-4
    for dim in dims:
        block_dim = int(dim/4)
        (D_Q_cds, Dinv_cds, I_1_cds, I_2_cds,
            A_1_cds, A_T_1_cds, A_2_cds, A_T_2_cds) = A_coords(dim)

        # Generate initial data
        D = np.eye(block_dim)
        Q = 0.5*np.eye(block_dim)
        Qinv = np.linalg.inv(Q)
        C = 2*np.eye(block_dim)
        B = np.eye(block_dim)
        E = np.eye(block_dim)
        #B = np.eye(block_dim)
        def obj(X):
            return A_dynamics(X, block_dim, C, B, E, Qinv)
        def grad_obj(X):
            return grad_A_dynamics(X, block_dim, C, B, E, Qinv)
        for i in range(N_rand):
            X = np.random.rand(dim, dim)
            val = obj(X)
            grad = grad_obj(X)
            num_grad = numerical_derivative(obj, X, eps)
            diff = np.sum(np.abs(grad - num_grad))
            print "X:\n", X
            print "grad:\n", grad
            print "num_grad:\n", num_grad
            print "diff: ", diff
            assert diff < tol
