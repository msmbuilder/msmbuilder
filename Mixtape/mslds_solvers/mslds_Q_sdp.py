from cvxopt import matrix, solvers, spmatrix, spdiag
from numpy import bmat, zeros, reshape, array, dot, shape, eye, shape, real
from numpy import ones
from numpy.linalg import pinv, eig
from scipy.linalg import block_diag, sqrtm
import scipy
import scipy.linalg
import numpy as np
import IPython

def construct_primal_matrix(x_dim, s, Z, F, Q, D, A):
    # x = [s vec(Z) vec(Q)]
    # F = B^{.5}
    # min c = s*dim + Tr Z
    # ------------------------
    #|Z+sI   F
    #|F.T    Q
    #|           D-Q   A
    #|           A.T D^{-1}
    #|                      Q
    #|                        Z
    # ------------------------
    P1 = zeros((2*x_dim, 2*x_dim))
    P1[:x_dim, :x_dim] = Z + s * eye(x_dim)
    P1[:x_dim, x_dim:] = F
    P1[x_dim:, :x_dim] = F.T
    P1[x_dim:, x_dim:] = Q
    print "eig(P1)"
    print eig(P1)[0]

    P2 = zeros((2*x_dim, 2*x_dim))
    P2[:x_dim, :x_dim] = D - Q
    P2[:x_dim, x_dim:] = A
    P2[x_dim:, :x_dim] = A.T
    Dinv = pinv(D)
    # To preserve symmetricity
    Dinv = (Dinv + Dinv.T)/2.
    P2[x_dim:, x_dim:] = Dinv
    # Add this small offset in hope of correcting for the numerical
    # errors in pinv
    eps = 1e-4
    P2 += eps * eye(2*x_dim)
    print "eig(P2)"
    print eig(P2)[0]

    P3 = zeros((x_dim, x_dim))
    P3[:,:] = Q
    print "eig(P3)"
    print eig(P3)[0]

    P4 = zeros((x_dim, x_dim))
    P4[:,:] = Z
    print "eig(P4)"
    print eig(P4)[0]

    P = scipy.linalg.block_diag(P1, P2, P3, P4)
    if min(np.linalg.eig(P)[0]) < 0:
        print "ERROR: P not PD in Q Solver!"
        IPython.embed()
    return P

def construct_coeff_matrix(x_dim, B):
    # x = [s vec(Z) vec(Q)]
    # F = B^{.5}
    # ------------------------
    #|Z+sI   F
    #|F.T    Q
    #|           D-Q   A
    #|           A.T D^{-1}
    #|                      Q
    #|                        Z
    # ------------------------

    g_dim = 6 * x_dim
    p_dim = 1 + 2* x_dim * (x_dim + 1) / 2
    G = spmatrix([], [], [], (g_dim**2, p_dim), 'd')
    # Block Matrix 1
    # First Block Column
    g1_dim = 2 * x_dim
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
    # Block Matrix 2
    g2_dim = 2 * x_dim
    # Third Block Column
    left = 0 * x_dim+g1_dim
    top = 0 * x_dim+g1_dim
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
    # Block Matrix 3
    g3_dim = x_dim
    # Fifth Block Column
    # Q
    left = 0 * x_dim+g1_dim+g2_dim
    top = 0 * x_dim+g1_dim+g2_dim
    prev = 1 + x_dim * (x_dim + 1) / 2
    for j in range(x_dim):  # cols
        for i in range(x_dim):  # rows
            mat_pos = left * g_dim + j * g_dim + top + i
            if i >= j:
                (i, j) = (j, i)
            vec_pos = prev + j * (j + 1) / 2 + i  # pos in param vector
            G[mat_pos, vec_pos] += 1.
    # Block Matrix 4
    g4_dim = x_dim
    # Sixth Block Column
    # Z
    left = 0 * x_dim+g1_dim+g2_dim+g3_dim
    top = 0 * x_dim+g1_dim+g2_dim+g3_dim
    prev = 1
    for j in range(x_dim):  # cols
        for i in range(x_dim):  # rows
            mat_pos = left * g_dim + j * g_dim + top + i
            if i >= j:
                (i, j) = (j, i)
            vec_pos = prev + j * (j + 1) / 2 + i  # pos in param vector
            G[mat_pos, vec_pos] += 1.
    Gs = [G]
    return Gs


def construct_const_matrix(x_dim, A, B, D):
    # F = B^{.5}
    # -----------------------
    #| 0      F
    #| F.T    0
    #|            D    A
    #|           A.T D^{-1}
    #|                       0
    #|                         0
    # -----------------------
    # Smallest number epsilon such that 1. + epsilon != 1.
    epsilon = np.finfo(np.float32).eps
    # Add a small positive offset to avoid taking sqrt of singular matrix
    F = real(sqrtm(B+epsilon*eye(x_dim)))
    # Construct B1
    H1 = zeros((2 * x_dim, 2 * x_dim))
    H1[:x_dim, x_dim:] = F
    H1[x_dim:, :x_dim] = F.T
    #eps = 1e-4
    #H1[x_dim:,x_dim:] = eps * eye(x_dim)

    #H1[:x_dim, :x_dim] = dot(F, dot(pinv(D - dot(A, dot(D, A.T))), F)) + 10 * eye(x_dim)
    #H1[x_dim:, x_dim:] = D - dot(A, dot(D, A.T))
    #print "eig(H1)+10K eye(x_dim)"
    #print eig(H1)[0]
    #print "eig(dot(F, dot(pinv(D), F)) + 10 * eye(x_dim))"
    #print eig(dot(F, dot(pinv(D), F)) + 10 * eye(x_dim))[0]
    #H1[:x_dim, :x_dim] = 0.
    #H1[x_dim:, x_dim:] = 0.
    H1 = matrix(H1)

    # Construct B2
    H2 = zeros((2 * x_dim, 2 * x_dim))
    H2[:x_dim, :x_dim] = D
    H2[:x_dim, x_dim:] = A
    H2[x_dim:, :x_dim] = A.T
    Dinv = pinv(D)
    # For symmmetricity
    Dinv = (Dinv + Dinv.T)/2.
    H2[x_dim:, x_dim:] = Dinv
    # Add this small offset in hope of correcting for the numerical
    # errors in pinv
    eps = 1e-4
    H2 += eps * eye(2*x_dim)
    H2 = matrix(H2)

    # Construct B3
    #min_epsilon = np.finfo(np.float32).eps
    #eps = max(0.1*min(eig(D)[0]), min_epsilon) # 0.5 should be param
    H3 = zeros((x_dim, x_dim))
    #H3 = -eps * eye(x_dim)
    #H3[:,:] = D[:,:]
    #H3 *= -eps
    H3 = matrix(H3)

    # Construct B4
    H4 = zeros((x_dim, x_dim))
    H4 = matrix(H4)

    # Construct Block matrix
    H = spdiag([H1,H2,H3,H4])
    hs = [H]
    return hs, F


def solve_Q(x_dim, A, B, D):
    # x = [s vec(Z) vec(Q)]
    print "SOLVE_Q!"
    print "eig(D)"
    print eig(D)[0]
    print "eig(dot(A, dot(D, A.T)))"
    print eig(dot(A, dot(D, A.T)))[0]
    print "eig(D - dot(A, dot(D, A.T)))"
    print eig(D- dot(A, dot(D, A.T)))[0]
    epsilon = np.finfo(np.float32).eps
    F = real(sqrtm(B+epsilon*eye(x_dim)))
    print "eig(F)"
    print eig(F)[0]
    print "F == F.T"
    print (abs(F - F.T) < 1e-3).all()
    print "max eig(A)"
    print max([np.linalg.norm(el) for el in eig(A)[0]])
    MAX_ITERS=400
    c_dim = 1 + 2 * x_dim * (x_dim + 1) / 2
    c = zeros((c_dim,1))
    # c = s*dim + Tr Z
    c[0] = x_dim
    prev = 1
    for i in range(x_dim):
        vec_pos = prev + i * (i + 1) / 2 + i
        c[vec_pos] = 1.
    cm = matrix(c)

    # Scale objective down by T for numerical stability
    eigs = eig(B)[0]
    T = max(abs(max(eigs)), abs(min(eigs)))
    Bdown = B / T
    Gs = construct_coeff_matrix(x_dim, Bdown)
    for i in range(len(Gs)):
        Gs[i] = -Gs[i]
    G = np.copy(matrix(Gs[0]))

    hs, _ = construct_const_matrix(x_dim, A, Bdown, D)
    for i in range(len(hs)):
        hs[i] = matrix(hs[i])
    # Smallest number epsilon such that 1. + epsilon != 1.
    epsilon = np.finfo(np.float32).eps
    # Add a small positive offset to avoid taking sqrt of singular matrix
    F = real(sqrtm(Bdown+epsilon*eye(x_dim)))

    # Construct primalstart
    Qprim = 0.99*(D - dot(A, dot(D, A.T)))
    Qpriminv = pinv(Qprim)
    Qpriminv = (Qpriminv + Qpriminv.T)/2.
    Zprim = dot(F, dot(Qpriminv, F.T)) + eye(x_dim)
    # pinv doesn't explicitly preserve symmetric matrices
    Zprim = (Zprim + Zprim.T)/2.
    sprim = max(eig(Zprim)[0])
    #IPython.embed()
    P = construct_primal_matrix(x_dim, sprim, Zprim, F, Qprim, D, A)
    primalstart = {}
    x = zeros((c_dim,1))
    x[0] = sprim
    count = 1
    for i in range(x_dim):
        for j in range(i+1):
            x[count] = Zprim[i,j]
            count += 1
    for i in range(x_dim):
        for j in range(i+1):
            x[count] = Qprim[i,j]
            count += 1
    x = matrix(x)
    primalstart['x'] = x
    primalstart['ss'] = [matrix(P)]
    #IPython.embed()

    solvers.options['maxiters'] = MAX_ITERS
    #solvers.options['debug'] = True
    sol = solvers.sdp(cm, Gs=Gs, hs=hs, primalstart=primalstart)
    print sol
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
