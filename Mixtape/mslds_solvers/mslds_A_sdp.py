from cvxopt import matrix, solvers, spmatrix
from numpy import bmat, zeros, reshape, array, dot, eye, outer, shape
from numpy import sqrt, real, ones
from numpy.linalg import pinv, eig, matrix_rank
from scipy.linalg import block_diag, sqrtm, pinv2
import numpy as np


def construct_coeff_matrix(x_dim, Q, C, B, E):
    # x = [s vec(Z) vec(A)]
    # F = Q^{-.5}(C-B) (not(!) symmetric)
    # J = Q^{-.5} (symmetric)
    # H = E^{.5} (symmetric)
    # Smallest number epsilon such that 1. + epsilon != 1.
    epsilon = np.finfo(np.float32).eps
    p_dim = 1 + x_dim * (x_dim + 1) / 2 + x_dim ** 2
    # Block Matrix 1
    g1_dim = 2 * x_dim
    #G1 = zeros((g1_dim ** 2, p_dim))
    G1 = {}
    G1_size = ((g1_dim ** 2, p_dim))
    # Add a small positive offset to avoid taking sqrt of singular matrix
    #J = real(sqrtm(pinv(Q)+epsilon*eye(x_dim)))
    J = real(sqrtm(pinv2(Q)+epsilon*eye(x_dim)))
    H = real(sqrtm(E+epsilon*eye(x_dim)))
    F = dot(J, C - B)
    # ------------------------------------------
    #|Z+sI-JAF.T -FA.TJ  JAH
    #|    (JAH).T         I
    #|                       D-eps_I    A
    #|                       A.T        D^{-1}
    #|                                         I  A.T
    #|                                         A   I
    #|                                                Z
    # -------------------------------------------
    # First Block Column
    # Z+sI-JAF.T -FA.TJ
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
            if (mat_pos, vec_pos) in G1:
                G1[(mat_pos, vec_pos)] += 1.
            else:
                G1[(mat_pos, vec_pos)] = 1.
    # sI
    prev = 0
    for i in range(x_dim):  # row/col on diag
        vec_pos = prev  # pos in param vector
        mat_pos = left * g1_dim + i * g1_dim + top + i
        if (mat_pos, vec_pos) in G1:
            G1[(mat_pos, vec_pos)] += 1.
        else:
            G1[(mat_pos, vec_pos)] = 1.
    # - J A F.T
    prev = 1 + x_dim * (x_dim + 1) / 2
    for i in range(x_dim):
        for j in range(x_dim):
            mat_pos = left * g1_dim + j * g1_dim + top + i
            # For (i,j)-th element in matrix M
            # do summation:
            #   M    = J A F.T
            #   M_ij = sum_m (JA)_im (F.T)_mj
            #        = sum_m (JA)_im F_jm
            #        = sum_m (sum_n J_in A_nm) F_jm
            #        = sum_m sum_n J_in A_nm F_jm
            for m in range(x_dim):
                for n in range(x_dim):
                    vec_pos = prev + n * x_dim + m
                    if (mat_pos, vec_pos) in G1:
                        G1[(mat_pos, vec_pos)] += -J[i, n] * F[j, m]
                    else:
                        G1[(mat_pos, vec_pos)] = -J[i, n] * F[j, m]
    # - F A.T J
    prev = 1 + x_dim * (x_dim + 1) / 2
    for i in range(x_dim):
        for j in range(x_dim):
            mat_pos = left * g1_dim + j * g1_dim + top + i
            # For (i,j)-th element in matrix M
            # do summation:
            #   M    = F A.T J
            #   M_ij = sum_m (FA.T)_im J_mj
            #        = sum_m (sum_n F_in (A.T)_nm) J_mj
            #        = sum_m (sum_n F_in A_mn) J_mj
            #        = sum_m sum_n F_in A_mn J_mj
            for m in range(x_dim):
                for n in range(x_dim):
                    vec_pos = prev + m * x_dim + n
                    if (mat_pos, vec_pos) in G1:
                        G1[(mat_pos, vec_pos)] += -F[i, n] * J[m, j]
                    else:
                        G1[(mat_pos, vec_pos)] = -F[i, n] * J[m, j]
    # H A.T J
    left = 0
    top = x_dim
    prev = 1 + x_dim * (x_dim + 1) / 2
    for i in range(x_dim):
        for j in range(x_dim):
            mat_pos = left * g1_dim + j * g1_dim + top + i
            # For (i,j)-th element in matrix M
            # do summation:
            #   M    = H A.T J
            #   M_ij = sum_m (HA.T)_im J_mj
            #        = sum_m (sum_n H_in (A.T)_nm) J_mj
            #        = sum_m (sum_n H_in A_mn) J_mj
            #        = sum_m sum_n H_in A_mn J_mj
            for m in range(x_dim):
                for n in range(x_dim):
                    vec_pos = prev + m * x_dim + n
                    if (mat_pos, vec_pos) in G1:
                        G1[(mat_pos, vec_pos)] += H[i, n] * J[m, j]
                    else:
                        G1[(mat_pos, vec_pos)] = H[i, n] * J[m, j]
    # Second Block Column
    # J A H
    left = x_dim
    top = 0
    prev = 1 + x_dim * (x_dim + 1) / 2
    for i in range(x_dim):
        for j in range(x_dim):
            mat_pos = left * g1_dim + j * g1_dim + top + i
            # For (i,j)-th element in matrix M
            # do summation:
            #   M    = J A H
            #   M_ij = sum_m (JA)_im H_mj
            #        = sum_m (JA)_im H_mj
            #        = sum_m (sum_n J_in A_nm) H_mj
            #        = sum_m sum_n J_in A_nm H_mj
            for m in range(x_dim):
                for n in range(x_dim):
                    vec_pos = prev + n * x_dim + m
                    if (mat_pos, vec_pos) in G1:
                        G1[(mat_pos, vec_pos)] += J[i, n] * H[m, j]
                    else:
                        G1[(mat_pos, vec_pos)] = J[i, n] * H[m, j]
    G1_I = [pair[0] for pair in G1.keys()]
    G1_J = [pair[1] for pair in G1.keys()]
    G1_x = G1.values()
    G1_mat = spmatrix(G1_x, G1_I, G1_J, G1_size)
    # Block Matrix 2
    g2_dim = 2 * x_dim
    G2_size = (g2_dim ** 2, p_dim)
    G2 = {}
    # Third Block Column
    # A.T
    left = 0 * x_dim
    top = 1 * x_dim
    prev = 1 + x_dim * (x_dim + 1) / 2
    for j in range(x_dim):  # cols
        for i in range(x_dim):  # rows
            vec_pos = prev + i * x_dim + j  # pos in param vector
            mat_pos = left * g2_dim + j * g2_dim + top + i
            if (mat_pos, vec_pos) in G2:
                G2[(mat_pos, vec_pos)] += 1
            else:
                G2[(mat_pos, vec_pos)] = 1
    # Fourth Block Column
    # A
    left = 1 * x_dim
    top = 0 * x_dim
    prev = 1 + x_dim * (x_dim + 1) / 2
    for j in range(x_dim):  # cols
        for i in range(x_dim):  # rows
            vec_pos = prev + j * x_dim + i  # pos in param vector
            mat_pos = left * g2_dim + j * g2_dim + top + i
            if (mat_pos, vec_pos) in G2:
                G2[(mat_pos, vec_pos)] += 1
            else:
                G2[(mat_pos, vec_pos)] = 1
    G2_I = [pair[0] for pair in G2.keys()]
    G2_J = [pair[1] for pair in G2.keys()]
    G2_x = G2.values()
    G2_mat = spmatrix(G2_x, G2_I, G2_J, G2_size)
    # Block Matrix 3
    g3_dim = 2 * x_dim
    G3_size = (g3_dim ** 2, p_dim)
    G3 = {}
    # Fifth Block Column
    # A
    left = 0 * x_dim
    top = 1 * x_dim
    prev = 1 + x_dim * (x_dim + 1) / 2
    for j in range(x_dim):  # cols
        for i in range(x_dim):  # rows
            vec_pos = prev + j * x_dim + i  # pos in param vector
            mat_pos = left * g3_dim + j * g3_dim + top + i
            if (mat_pos, vec_pos) in G3:
                G3[(mat_pos, vec_pos)] += 1
            else:
                G3[(mat_pos, vec_pos)] = 1
    # Sixth Block Column
    # A.T
    left = 1 * x_dim
    top = 0 * x_dim
    prev = 1 + x_dim * (x_dim + 1) / 2
    for j in range(x_dim):  # cols
        for i in range(x_dim):  # rows
            vec_pos = prev + i * x_dim + j  # pos in param vector
            mat_pos = left * g3_dim + j * g3_dim + top + i
            if (mat_pos, vec_pos) in G3:
                G3[(mat_pos, vec_pos)] += 1
            else:
                G3[(mat_pos, vec_pos)] = 1
    G3_I = [pair[0] for pair in G3.keys()]
    G3_J = [pair[1] for pair in G3.keys()]
    G3_x = G3.values()
    G3_mat = spmatrix(G3_x, G3_I, G3_J, G3_size)
    # Block Matrix 4
    g4_dim = 1 * x_dim
    G4_size = (g4_dim ** 2, p_dim)
    G4 = {}
    # Seventh Block Column
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
            if (mat_pos, vec_pos) in G4:
                G4[(mat_pos, vec_pos)] += 1
            else:
                G4[(mat_pos, vec_pos)] = 1
    G4_I = [pair[0] for pair in G4.keys()]
    G4_J = [pair[1] for pair in G4.keys()]
    G4_x = G4.values()
    G4_mat = spmatrix(G4_x, G4_I, G4_J, G4_size)

    Gs = [G1_mat, G2_mat, G3_mat, G4_mat]
    return Gs, F, J, H

def construct_const_matrix(x_dim, Q, D):
    # --------------------------
    #| 0   0
    #| 0   I
    #|        D-eps_I    0
    #|         0        D^{-1}
    #|                         I  0
    #|                         0  I
    #|                              0
    # --------------------------
    # Construct B1
    H1 = zeros((2 * x_dim, 2 * x_dim))
    H1[x_dim:, x_dim:] = eye(x_dim)

    # Construct B2
    eps = 1e-4
    H2 = zeros((2 * x_dim, 2 * x_dim))
    H2[:x_dim, :x_dim] = D - eps * eye(x_dim)
    H2[x_dim:, x_dim:] = pinv(D)

    # Construct B3
    H3 = eye(2*x_dim)

    # Construct B5
    H4 = zeros((x_dim, x_dim))

    # Construct Block matrix
    hs = [H1, H2, H3, H4]
    return hs


def solve_A(x_dim, B, C, E, D, Q):
    # x = [s vec(Z) vec(A)]
    MAX_ITERS = 30
    c_dim = 1 + x_dim * (x_dim + 1) / 2 + x_dim ** 2
    c = zeros(c_dim)
    c[0] = x_dim
    prev = 1
    for i in range(x_dim):
        vec_pos = prev + i * (i + 1) / 2 + i
        c[vec_pos] = 1.
    cm = matrix(c)

    Gs, _, _, _ = construct_coeff_matrix(x_dim, Q, C, B, E)
    for i in range(len(Gs)):
        Gs[i] = -Gs[i]

    hs = construct_const_matrix(x_dim, Q, D)
    for i in range(len(hs)):
        hs[i] = matrix(hs[i])

    solvers.options['maxiters'] = MAX_ITERS
    sol = solvers.sdp(cm, Gs=Gs, hs=hs)
    return sol, c, Gs, hs


def test_A_generate_constraints(x_dim):
    # Define constants
    xs = zeros((2, x_dim))
    xs[0] = ones(x_dim)
    xs[1] = 2 * ones(x_dim)
    b = 0.5 * ones((x_dim, 1))
    Q = eye(x_dim)
    D = 2 * eye(x_dim)
    B = outer(xs[1], xs[0])
    E = outer(xs[0], xs[0])
    C = outer(b, xs[0])
    return B, C, E, D, Q


def test_A_solve_sdp(x_dim):
    B, C, E, D, Q = test_A_generate_constraints(x_dim)
    sol, c, G, h = solve_A(x_dim, B, C, E, D, Q)
    return sol, c, G, h, B, C, E, D, Q
