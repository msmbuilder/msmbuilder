from cvxopt import matrix, solvers
from numpy import bmat, zeros, reshape, array, dot, eye, outer, shape
from numpy import sqrt, real, ones
from numpy.linalg import pinv, eig, matrix_rank
from scipy.linalg import block_diag, sqrtm


def construct_coeff_matrix(x_dim, Q, C, B, E):
    # x = [s vec(Z) vec(A)]
    # F = Q^{-.5}(C-B) (not(!) symmetric)
    # J = Q^{-.5} (symmetric)
    # H = E^{.5} (symmetric)
    #g_dim = 7 * x_dim + 1
    g_dim = 7 * x_dim
    G = zeros((g_dim ** 2, 1 + x_dim * (x_dim + 1) / 2 + x_dim ** 2))
    J = real(sqrtm(pinv(Q)))
    H = real(sqrtm(E))
    F = dot(J, C - B)
    print "F"
    print F
    print "J"
    print J
    print "H"
    print H
    # k = x_dim
    # ------------------------------------------
    #|Z+sI-JAF.T -FA.TJ  JAH
    #|    (JAH).T         I
    #|                       D-Q+eps_I    A
    #|                       A.T        D^{-1}
    #|                                         I  A.T
    #|                                         A   I
    #|                                                Z
    #|//                                                   s
    # -------------------------------------------
    # First Block Column
    # Z+sI-JAF.T -FA.TJ
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
        vec_pos = prev  # pos in param vector
        mat_pos = left * g_dim + i * g_dim + top + i
        G[mat_pos, vec_pos] += 1.
    # - J A F.T
    prev = 1 + x_dim * (x_dim + 1) / 2
    for i in range(x_dim):
        for j in range(x_dim):
            mat_pos = left * g_dim + j * g_dim + top + i
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
                    G[mat_pos, vec_pos] += -J[i, n] * F[j, m]
    # - F A.T J
    prev = 1 + x_dim * (x_dim + 1) / 2
    for i in range(x_dim):
        for j in range(x_dim):
            mat_pos = left * g_dim + j * g_dim + top + i
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
                    G[mat_pos, vec_pos] += -F[i, n] * J[m, j]
    # H A.T J
    left = 0
    top = x_dim
    prev = 1 + x_dim * (x_dim + 1) / 2
    for i in range(x_dim):
        for j in range(x_dim):
            mat_pos = left * g_dim + j * g_dim + top + i
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
                    G[mat_pos, vec_pos] += H[i, n] * J[m, j]
    # Second Block Column
    # J A H
    left = x_dim
    top = 0
    prev = 1 + x_dim * (x_dim + 1) / 2
    for i in range(x_dim):
        for j in range(x_dim):
            mat_pos = left * g_dim + j * g_dim + top + i
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
                    G[mat_pos, vec_pos] += J[i, n] * H[m, j]
    # Third Block Column
    # A.T
    left = 2 * x_dim
    top = 3 * x_dim
    prev = 1 + x_dim * (x_dim + 1) / 2
    for j in range(x_dim):  # cols
        for i in range(x_dim):  # rows
            vec_pos = prev + i * x_dim + j  # pos in param vector
            mat_pos = left * g_dim + j * g_dim + top + i
            G[mat_pos, vec_pos] += 1.
    # Fourth Block Column
    # A
    left = 3 * x_dim
    top = 2 * x_dim
    prev = 1 + x_dim * (x_dim + 1) / 2
    for j in range(x_dim):  # cols
        for i in range(x_dim):  # rows
            vec_pos = prev + j * x_dim + i  # pos in param vector
            mat_pos = left * g_dim + j * g_dim + top + i
            G[mat_pos, vec_pos] += 1.
    # ------------------
    # Fifth Block Column
    # A
    left = 4 * x_dim
    top = 5 * x_dim
    prev = 1 + x_dim * (x_dim + 1) / 2
    for j in range(x_dim):  # cols
        for i in range(x_dim):  # rows
            vec_pos = prev + j * x_dim + i  # pos in param vector
            mat_pos = left * g_dim + j * g_dim + top + i
            G[mat_pos, vec_pos] += 1.
    # Sixth Block Column
    # A.T
    left = 5 * x_dim
    top = 4 * x_dim
    prev = 1 + x_dim * (x_dim + 1) / 2
    for j in range(x_dim):  # cols
        for i in range(x_dim):  # rows
            vec_pos = prev + i * x_dim + j  # pos in param vector
            mat_pos = left * g_dim + j * g_dim + top + i
            G[mat_pos, vec_pos] += 1.
    # Seventh Block Column
    # Z
    left = 6 * x_dim
    top = 6 * x_dim
    prev = 1
    for j in range(x_dim):  # cols
        for i in range(x_dim):  # rows
            mat_pos = left * g_dim + j * g_dim + top + i
            if i >= j:
                (i, j) = (j, i)
            vec_pos = prev + j * (j + 1) / 2 + i  # pos in param vector
            G[mat_pos, vec_pos] += 1.

    ## s
    #left = 7 * x_dim
    #top = 7 * x_dim
    #prev = 0
    #mat_pos = left * g_dim + top
    #vec_pos = 0
    #G[mat_pos, vec_pos] += 1.
    return G, F, J, H


def construct_const_matrix(x_dim, Q, D):
    # --------------------------
    #| 0   0
    #| 0   I
    #|        D+eps_I-Q    0
    #|//        D+eps_I    0
    #|         0        D^{-1}
    #|                         I
    #|                            I
    #|                              0
    # --------------------------
    # Construct B1
    B1 = zeros((2 * x_dim, 2 * x_dim))
    B1[x_dim:, x_dim:] = eye(x_dim)

    # Construct B2
    eps = 1e-3
    B2 = zeros((2 * x_dim, 2 * x_dim))
    B2[:x_dim, :x_dim] = D - Q + eps * eye(x_dim)
    B2[:x_dim, :x_dim] = D + eps * eye(x_dim)
    B2[x_dim:, x_dim:] = pinv(D)

    # Construct B3
    B3 = eye(x_dim)

    # Construct B4
    B4 = eye(x_dim)

    # Construct B5
    B5 = zeros((x_dim, x_dim))

    ## Construct B6
    #B6 = zeros((1, 1))

    # Construct Block matrix
    #h = block_diag(B1, B2, B3, B4, B5, B6)
    h = block_diag(B1, B2, B3, B4, B5)
    return h


def solve_A(x_dim, B, C, E, D, Q):
    # x = [s vec(Z) vec(A)]
    print "Q:"
    print Q
    print "D:"
    print D
    print "B:"
    print B
    print "C:"
    print C
    print "E:"
    print E
    MAX_ITERS = 30
    c_dim = 1 + x_dim * (x_dim + 1) / 2 + x_dim ** 2
    c = zeros(c_dim)
    c[0] = x_dim
    prev = 1
    for i in range(x_dim):
        vec_pos = prev + i * (i + 1) / 2 + i
        c[vec_pos] = 1.
    print "c:"
    print c
    cm = matrix(c)

    G, _, _, _ = construct_coeff_matrix(x_dim, Q, C, B, E)
    G = -G  # set negative since s = h - Gx in cvxopt's sdp solver
    Gs = [matrix(G)]

    h = construct_const_matrix(x_dim, Q, D)
    hs = [matrix(h)]

    solvers.options['maxiters'] = MAX_ITERS
    sol = solvers.sdp(cm, Gs=Gs, hs=hs)
    print "A-solution"
    print sol['x']
    return sol, c, G, h


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
