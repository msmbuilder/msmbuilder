from cvxopt import matrix, solvers
from numpy import bmat, zeros, reshape, array, dot, shape, eye, shape, real
from numpy import ones
from numpy.linalg import pinv, eig
from scipy.linalg import block_diag, sqrtm
import numpy as np


def construct_coeff_matrix(x_dim, B):
    # x = [s vec(Z) vec(Q)]
    # F = B^{.5}
    # ------------------------
    #|Z+sI  F
    #| F    Q
    #|           D-Q   A
    #|           A.T D^{-1}
    #|                      Q
    #|                        Z
    # ------------------------

    # Block Matrix 1
    # First Block Column
    g1_dim = 2 * x_dim
    G1 = zeros((g1_dim ** 2, 1 + 2 * x_dim * (x_dim + 1) / 2))
    # Z + sI
    left = 0
    top = 0
    # Z
    prev = 1
    for j in range(x_dim):  # cols
        for i in range(x_dim):  # rows
            mat_pos = left * g1_dim + j * g1_dim + top + i
            if i >= j:
                (i, j) = (j, i)
            vec_pos = prev + j * (j + 1) / 2 + i  # pos in param vector
            G1[mat_pos, vec_pos] += 1.
    # sI
    prev = 0
    for i in range(x_dim):  # row/col on diag
        vec_pos = 0  # pos in param vector
        mat_pos = left * g1_dim + i * g1_dim + top + i
        G1[mat_pos, vec_pos] += 1.
    # Second Block Column
    # Q
    left = x_dim
    top = x_dim
    prev = 1 + x_dim * (x_dim + 1) / 2
    for j in range(x_dim):  # cols
        for i in range(x_dim):  # rows
            mat_pos = left * g1_dim + j * g1_dim + top + i
            if i >= j:
                (i, j) = (j, i)
            vec_pos = prev + j * (j + 1) / 2 + i  # pos in param vector
            G1[mat_pos, vec_pos] += 1.
    # Block Matrix 2
    g2_dim = 2 * x_dim
    G2 = zeros((g2_dim ** 2, 1 + 2 * x_dim * (x_dim + 1) / 2))
    # Third Block Column
    # -Q
    left = 0 * x_dim
    top = 0 * x_dim
    prev = 1 + x_dim * (x_dim + 1) / 2
    for j in range(x_dim):  # cols
        for i in range(x_dim):  # rows
            mat_pos = left * g2_dim + j * g2_dim + top + i
            if i >= j:
                (i, j) = (j, i)
            vec_pos = prev + j * (j + 1) / 2 + i  # pos in param vector
            G2[mat_pos, vec_pos] += -1.
    # Fourth Block Column
    # -------------------
    # Block Matrix 3
    g3_dim = x_dim
    G3 = zeros((g3_dim ** 2, 1 + 2 * x_dim * (x_dim + 1) / 2))
    # Fifth Block Column
    # Q
    left = 0 * x_dim
    top = 0 * x_dim
    prev = 1 + x_dim * (x_dim + 1) / 2
    for j in range(x_dim):  # cols
        for i in range(x_dim):  # rows
            mat_pos = left * g3_dim + j * g3_dim + top + i
            if i >= j:
                (i, j) = (j, i)
            vec_pos = prev + j * (j + 1) / 2 + i  # pos in param vector
            G3[mat_pos, vec_pos] += 1.
    # Block Matrix 4
    g4_dim = x_dim
    G4 = zeros((g4_dim ** 2, 1 + 2 * x_dim * (x_dim + 1) / 2))
    # Sixth Block Column
    # Z
    left = 0 * x_dim
    top = 0 * x_dim
    prev = 1
    for j in range(x_dim):  # cols
        for i in range(x_dim):  # rows
            mat_pos = left * g4_dim + j * g4_dim + top + i
            if i >= j:
                (i, j) = (j, i)
            vec_pos = prev + j * (j + 1) / 2 + i  # pos in param vector
            G4[mat_pos, vec_pos] += 1.
    Gs = [G1, G2, G3, G4]
    return Gs


def construct_const_matrix(x_dim, A, B, D):
    # F = B^{.5}
    # -----------------------
    #| 0    F
    #| F    0
    #|           D    A
    #|          A.T D^{-1}
    #|                      0
    #|                        0
    # -----------------------
    # Smallest number epsilon such that 1. + epsilon != 1.
    epsilon = np.finfo(np.float32).eps
    # Add a small positive offset to avoid taking sqrt of singular matrix
    F = real(sqrtm(B+epsilon*eye(x_dim)))
    # Construct B1
    H1 = zeros((2 * x_dim, 2 * x_dim))
    H1[x_dim:, :x_dim] = F
    H1[:x_dim, x_dim:] = F

    # Construct B2
    H2 = zeros((2 * x_dim, 2 * x_dim))
    H2[:x_dim, :x_dim] = D
    H2[:x_dim, x_dim:] = A
    H2[x_dim:, :x_dim] = A.T
    H2[x_dim:, x_dim:] = pinv(D)

    # Construct B3
    H3 = zeros((x_dim, x_dim))

    # Construct B4
    H4 = zeros((x_dim, x_dim))

    # Construct Block matrix
    #h = block_diag(B1, B2, B3, B4)
    hs = [H1, H2, H3, H4]
    return hs, F


def solve_Q(x_dim, A, B, D):
    # x = [s vec(Z) vec(Q)]
    c_dim = 1 + 2 * x_dim * (x_dim + 1) / 2
    c = zeros(c_dim)
    c[0] = x_dim
    prev = 1
    for i in range(x_dim):
        vec_pos = prev + i * (i + 1) / 2 + i
        c[vec_pos] = 1
    cm = matrix(c)

    Gs = construct_coeff_matrix(x_dim, B)
    for i in range(len(Gs)):
        Gs[i] = matrix(-Gs[i])

    hs, _ = construct_const_matrix(x_dim, A, B, D)
    for i in range(len(hs)):
        hs[i] = matrix(hs[i])

    sol = solvers.sdp(cm, Gs=Gs, hs=hs)
    return sol, c, Gs, hs


def test_Q_generate_constraints(x_dim):
    # Define constants
    xs = zeros((2, x_dim))
    xs[0] = ones(x_dim)
    xs[1] = ones(x_dim)
    b = 0.5 * ones((x_dim, 1))
    A = 0.9 * eye(x_dim)
    D = 2 * eye(x_dim)
    v = reshape(xs[1], (x_dim, 1)) - dot(A, reshape(xs[0], (x_dim, 1))) - b
    v = reshape(v, (len(v), 1))
    B = dot(v, v.T)
    return A, B, D


def test_Q_solve_sdp(x_dim):
    A, B, D = test_Q_generate_constraints(x_dim)
    sol, c, G, h = solve_Q(x_dim, A, B, D)
    return sol, c, G, h, A, B, D
