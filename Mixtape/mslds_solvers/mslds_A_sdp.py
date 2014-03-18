from cvxopt import matrix, solvers, spmatrix, spdiag, sparse
from numpy import bmat, zeros, reshape, array, dot, eye, outer, shape
from numpy import sqrt, real, ones
from numpy.linalg import pinv, eig, matrix_rank
from scipy.linalg import block_diag, sqrtm, pinv2
import scipy
import scipy.linalg
import numpy as np
import cvxopt.misc as misc
import math
import IPython
import pdb

def construct_primal_matrix(x_dim, s, Z, A, Q, F, J, H, D):
    # x = [s vec(Z) vec(A)]
    # F = Q^{-.5}(C-B) (not(!) symmetric)
    # J = Q^{-.5} (symmetric)
    # H = E^{.5} (symmetric)
    # ------------------------------------------
    #|Z+sI-JAF.T -FA.TJ  JAH
    #|    (JAH).T         I
    #|                       D-eps D    A
    #|                       A.T        D^{-1}
    #|                                         I  A.T
    #|                                         A   I
    #|                                                Z
    # -------------------------------------------
    # Smallest number epsilon such that 1. + epsilon != 1.
    # Construct P1
    P1 = zeros((2 * x_dim, 2 * x_dim))
    UL = (Z + s * eye(x_dim) - dot(J,dot(A, F.T)) - dot(F, dot(A.T, J)))
    # Explicitly symmetrize due to numeric issues
    UL = (UL + UL.T)/2.
    P1[:x_dim, :x_dim] = UL
    P1[:x_dim, x_dim:] = dot(J, dot(A, H))
    P1[x_dim:, :x_dim] = dot(J, dot(A, H)).T
    P1[x_dim:, x_dim:] = eye(x_dim)
    P1 = matrix(P1)

    # Construct P2
    eps = 1e-4
    P2 = zeros((2 * x_dim, 2 * x_dim))
    P2[:x_dim, :x_dim] = (1 - eps) * D
    P2[:x_dim, x_dim:] = A
    P2[x_dim:, :x_dim] = A.T
    Dinv = pinv(D)
    # Explicitly symmetrize
    Dinv = (Dinv + Dinv.T)/2.
    P2[x_dim:, x_dim:] = Dinv
    P2 = matrix(P2)

    # Construct P3
    P3 = eye(2*x_dim)
    P3[:x_dim, x_dim:] = A.T
    P3[x_dim:, :x_dim] = A
    P3 = matrix(P3)

    # Construct P5
    P4 = zeros((x_dim, x_dim))
    P4[:,:] = Z
    P4 = matrix(P4)

    #IPython.embed()
    # Construct Block matrix
    P = scipy.linalg.block_diag(P1, P2, P3, P4)
    if min(np.linalg.eig(P)[0]) < 0:
        print "ERROR: P not PD in A Solver!"
        IPython.embed()
    return P

def construct_coeff_matrix(x_dim, Q, C, B, E):
    # x = [s vec(Z) vec(A)]
    # F = Q^{-.5}(C-B) (not(!) symmetric)
    # J = Q^{-.5} (symmetric)
    # H = E^{.5} (symmetric)
    # ------------------------------------------
    #|Z+sI-JAF.T -FA.TJ  JAH
    #|    (JAH).T         I
    #|                       D-eps_I    A
    #|                       A.T        D^{-1}
    #|                                         I  A.T
    #|                                         A   I
    #|                                                Z
    # -------------------------------------------
    # Smallest number epsilon such that 1. + epsilon != 1.
    epsilon = np.finfo(np.float32).eps
    p_dim = 1 + x_dim * (x_dim + 1) / 2 + x_dim ** 2
    g_dim = 7 * x_dim
    G = spmatrix([], [], [], (g_dim**2, p_dim), 'd')
    # Block Matrix 1
    g1_dim = 2 * x_dim
    # Add a small positive offset to avoid taking sqrt of singular matrix
    #J = real(sqrtm(pinv(Q)+epsilon*eye(x_dim)))
    J = real(sqrtm(pinv2(Q)+epsilon*eye(x_dim)))
    H = real(sqrtm(E+epsilon*eye(x_dim)))
    F = dot(J, C - B)
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
                (it, jt) = (j, i)
            else:
                (it, jt) = (i, j)
            vec_pos = prev + jt * (jt + 1) / 2 + it  # pos in param vector
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
            #   M    = -J A F.T
            #   M_ij = -sum_m (JA)_im (F.T)_mj
            #        = -sum_m (JA)_im F_jm
            #        = -sum_m (sum_n J_in A_nm) F_jm
            #        = -sum_m sum_n J_in A_nm F_jm
            for m in range(x_dim):
                for n in range(x_dim):
                    val = -J[i, n] * F[j, m]
                    vec_pos = prev + n * x_dim + m
                    G[mat_pos, vec_pos] += val
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
    # Block Matrix 2
    g2_dim = 2 * x_dim
    # Third Block Column
    # A.T
    left = 0 * x_dim + g1_dim
    top = 1 * x_dim + g1_dim
    prev = 1 + x_dim * (x_dim + 1) / 2
    for j in range(x_dim):  # cols
        for i in range(x_dim):  # rows
            vec_pos = prev + i * x_dim + j  # pos in param vector
            mat_pos = left * g_dim + j * g_dim + top + i
            G[mat_pos, vec_pos] += 1
    # Fourth Block Column
    # A
    left = 1 * x_dim + g1_dim
    top = 0 * x_dim + g1_dim
    prev = 1 + x_dim * (x_dim + 1) / 2
    for j in range(x_dim):  # cols
        for i in range(x_dim):  # rows
            vec_pos = prev + j * x_dim + i  # pos in param vector
            mat_pos = left * g_dim + j * g_dim + top + i
            G[mat_pos, vec_pos] += 1
    # Block Matrix 3
    g3_dim = 2 * x_dim
    # Fifth Block Column
    # A
    left = 0 * x_dim + g1_dim + g2_dim
    top = 1 * x_dim + g1_dim + g2_dim
    prev = 1 + x_dim * (x_dim + 1) / 2
    for j in range(x_dim):  # cols
        for i in range(x_dim):  # rows
            vec_pos = prev + j * x_dim + i  # pos in param vector
            mat_pos = left * g_dim + j * g_dim + top + i
            G[mat_pos, vec_pos] += 1
    # Sixth Block Column
    # A.T
    left = 1 * x_dim + g1_dim + g2_dim
    top = 0 * x_dim + g1_dim + g2_dim
    prev = 1 + x_dim * (x_dim + 1) / 2
    for j in range(x_dim):  # cols
        for i in range(x_dim):  # rows
            vec_pos = prev + i * x_dim + j  # pos in param vector
            mat_pos = left * g_dim + j * g_dim + top + i
            G[mat_pos, vec_pos] += 1
    # Block Matrix 4
    g4_dim = 1 * x_dim
    # Seventh Block Column
    # Z
    left = 0 * x_dim+g1_dim+g2_dim+g3_dim
    top = 0 * x_dim+g1_dim+g2_dim+g3_dim
    prev = 1
    for j in range(x_dim):  # cols
        for i in range(x_dim):  # rows
            mat_pos = left * g_dim + j * g_dim + top + i
            if i >= j:
                (it, jt) = (j, i)
            else:
                (it, jt) = (i, j)
            vec_pos = prev + jt * (jt + 1) / 2 + it  # pos in param vector
            G[mat_pos, vec_pos] += 1

    Gs = [G]
    Z = matrix(zeros((p_dim, p_dim)))
    I = matrix(eye(g_dim**2))
    D = matrix(sparse([[Z, G], [G.T, -I]]))
    return Gs, F, J, H

def construct_const_matrix(x_dim, D):
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
    H1 = matrix(H1)

    # Construct B2
    eps = 1e-4
    H2 = zeros((2 * x_dim, 2 * x_dim))
    H2[:x_dim, :x_dim] = D - eps * D
    H2[x_dim:, x_dim:] = pinv(D)
    H2 = matrix(H2)

    # Construct B3
    H3 = eye(2*x_dim)
    H3 = matrix(H3)

    # Construct B5
    H4 = zeros((x_dim, x_dim))
    H4 = matrix(H4)

    # Construct Block matrix
    H = spdiag([H1,H2,H3,H4])
    hs = [H]
    return hs


def solve_A(x_dim, B, C, E, D, Q):
    # x = [s vec(Z) vec(A)]
    print "SOLVE_A!"
    MAX_ITERS = 100
    c_dim = 1 + x_dim * (x_dim + 1) / 2 + x_dim ** 2
    c = zeros((c_dim,1))
    c[0] = x_dim
    prev = 1
    for i in range(x_dim):
        vec_pos = prev + i * (i + 1) / 2 + i
        c[vec_pos] = 1.
    cm = matrix(c)

    # Scale objective down by T for numerical stability
    eigsQinv = max([abs(1./q) for q in eig(Q)[0]])
    eigsE = max([abs(e) for e in eig(E)[0]])
    eigsCB = max([abs(cb) for cb in eig(C-B)[0]])
    S = max(eigsQinv, eigsE, eigsCB)
    Qdown = Q / S
    Edown = E / S
    Cdown = C / S
    Bdown = B / S
    # Ensure that D doesn't have negative eigenvals
    # due to numerical issues
    min_D_eig = min(eig(D)[0])
    if min_D_eig < 0:
        # assume abs(min_D_eig) << 1
        D = D + 2 * abs(min_D_eig) * eye(x_dim)
    Gs, _, _, _ = construct_coeff_matrix(x_dim, Qdown, Cdown, Bdown, Edown)
    for i in range(len(Gs)):
        Gs[i] = -Gs[i] + 1e-6

    hs = construct_const_matrix(x_dim, D)

    epsilon = np.finfo(np.float32).eps
    eps = 1e-4
    # Add a small positive offset to avoid taking sqrt of singular matrix
    J = real(sqrtm(pinv2(Qdown)+epsilon*eye(x_dim)))
    H = real(sqrtm(Edown+epsilon*eye(x_dim)))
    F = dot(J, Cdown - Bdown)
    Aprim = 0.9 * (1 - eps) * eye(x_dim)
    Zprim = (dot(J, dot(Aprim, F.T)) + dot(F, dot(Aprim.T, J))
                + dot(dot(J, dot(Aprim, H)),dot(J, dot(Aprim, H)).T))
    min_eig = abs(min(eig(Zprim)[0]))
    Zprim += 2 * min_eig * eye(x_dim)
    # Explicitly symmetrize the matrix
    # (due to numerical issues)
    Zprim = (Zprim + Zprim.T)/2.
    sprim = 10
    P = construct_primal_matrix(x_dim, sprim, Zprim, Aprim, Qdown, F, J, H, D)
    primalstart = {}

    x = zeros((c_dim,1))
    x[0] = sprim
    count = 1
    for i in range(x_dim):
        for j in range(i+1):
            x[count] = Zprim[i,j]
            count += 1
    for i in range(x_dim):
        for j in range(x_dim):
            x[count] = Aprim[i,j]
            count += 1
    x = matrix(x)
    primalstart['x'] = x
    primalstart['ss'] = [matrix(P)]
    solvers.options['maxiters'] = MAX_ITERS
    sol = solvers.sdp(cm, Gs=Gs, hs=hs, primalstart=primalstart)
    print sol
    # check norm of A:
    avec = np.array(sol['x'])
    avec = avec[1 + x_dim * (x_dim + 1) / 2:]
    A = np.reshape(avec, (x_dim, x_dim), order='F')
    # Set this for debugging purposes
    #if max(eig(A)[0]) > 1:
    #    print "A NORM > 1!"
    #    pdb.set_trace()
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
