from cvxopt import matrix, solvers
from numpy import bmat, zeros, reshape, array, dot, shape, eye, shape, real
from numpy import ones
from numpy.linalg import pinv, eig
from scipy.linalg import block_diag, sqrtm


def construct_coeff_matrix(x_dim, B):
    # x = [s vec(Z) vec(Q)]
    # F = B^{.5}
    g_dim = 6 * x_dim
    G = zeros((g_dim ** 2, 1 + 2 * x_dim * (x_dim + 1) / 2))
    # -----------------------
    #|Z+sI  F
    #| F    Q
    #|           D-Q   A
    #|           A.T D^{-1}
    #|                      Q -epsilon I
    #|                                    Z
    # -----------------------

    # First Block Column
    # Z + sI
    left = 0
    top = 0
    # Z
    prev = 1
    for j in range(x_dim):  # cols
        for i in range(x_dim):  # rows
            mat_pos = left * g_dim + j * g_dim + top + i
            if i >= j:
                (i, j) = (j, i)
            vec_pos = prev + j * (j + 1) / 2 + i  # pos in param vector
            G[mat_pos, vec_pos] += 1.
    # sI
    prev = 0
    for i in range(x_dim):  # row/col on diag
        vec_pos = 0  # pos in param vector
        mat_pos = left * g_dim + i * g_dim + top + i
        G[mat_pos, vec_pos] += 1.
    # Second Block Column
    # Q
    left = x_dim
    top = x_dim
    prev = 1 + x_dim * (x_dim + 1) / 2
    for j in range(x_dim):  # cols
        for i in range(x_dim):  # rows
            mat_pos = left * g_dim + j * g_dim + top + i
            if i >= j:
                (i, j) = (j, i)
            vec_pos = prev + j * (j + 1) / 2 + i  # pos in param vector
            G[mat_pos, vec_pos] += 1.
    # Third Block Column
    # -Q
    left = 2 * x_dim
    top = 2 * x_dim
    prev = 1 + x_dim * (x_dim + 1) / 2
    for j in range(x_dim):  # cols
        for i in range(x_dim):  # rows
            mat_pos = left * g_dim + j * g_dim + top + i
            if i >= j:
                (i, j) = (j, i)
            vec_pos = prev + j * (j + 1) / 2 + i  # pos in param vector
            G[mat_pos, vec_pos] += -1.
    # Fourth Block Column
    # -------------------
    # Fifth Block Column
    # Q
    left = 4 * x_dim
    top = 4 * x_dim
    prev = 1 + x_dim * (x_dim + 1) / 2
    for j in range(x_dim):  # cols
        for i in range(x_dim):  # rows
            mat_pos = left * g_dim + j * g_dim + top + i
            if i >= j:
                (i, j) = (j, i)
            vec_pos = prev + j * (j + 1) / 2 + i  # pos in param vector
            G[mat_pos, vec_pos] += 1.
    # Sixth Block Column
    # Z
    left = 5 * x_dim
    top = 5 * x_dim
    prev = 1
    for j in range(x_dim):  # cols
        for i in range(x_dim):  # rows
            mat_pos = left * g_dim + j * g_dim + top + i
            if i >= j:
                (i, j) = (j, i)
            vec_pos = prev + j * (j + 1) / 2 + i  # pos in param vector
            G[mat_pos, vec_pos] += 1.
    return G


def construct_const_matrix(x_dim, A, B, D):
    # F = B^{.5}
    # -----------------------
    #| 0    F
    #| F    0
    #|           D    A
    #|          A.T D^{-1}
    #|                      0
    #|                        0
    #|                          0
    # -----------------------
    F = real(sqrtm(B))
    # Construct B1
    B1 = zeros((2 * x_dim, 2 * x_dim))
    B1[x_dim:, :x_dim] = F
    B1[:x_dim, x_dim:] = F

    # Construct B2
    eps = 1e-3
    B2 = zeros((2 * x_dim, 2 * x_dim))
    B2[:x_dim, :x_dim] = D
    B2[:x_dim, x_dim:] = A
    B2[x_dim:, :x_dim] = A.T
    B2[x_dim:, x_dim:] = pinv(D)

    # Construct B3
    B3 = zeros((x_dim, x_dim))

    # Construct B4
    B4 = zeros((x_dim, x_dim))

    # Construct Block matrix
    h = block_diag(B1, B2, B3, B4)
    return h, F


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

    G = construct_coeff_matrix(x_dim, B)
    G = -G  # set negative since s = h - Gx in cvxopt's sdp solver
    Gs = [matrix(G)]

    h, _ = construct_const_matrix(x_dim, A, B, D)
    hs = [matrix(h)]

    sol = solvers.sdp(cm, Gs=Gs, hs=hs)
    return sol, c, G, h


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
