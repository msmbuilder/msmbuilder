# Author:  Bharath Ramsundar <bharath.ramsundar@gmail.com>
# Contributors:
# Copyright (c) 2014, Stanford University
# All rights reserved.

# Mixtape is free software: you can redistribute it and/or modify
# it under the terms of the GNU Lesser General Public License as
# published by the Free Software Foundation, either version 2.1
# of the License, or (at your option) any later version.
#
# This library is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Lesser General Public License for more details.
#
# You should have received a copy of the GNU Lesser General Public
# License along with Mixtape. If not, see <http://www.gnu.org/licenses/>.

from __future__ import print_function, division, absolute_import

from cvxopt import matrix, solvers, spmatrix, spdiag
from numpy import bmat, zeros, reshape, array, dot, shape, eye, shape, real
from numpy import ones
from numpy.linalg import pinv, eig
from scipy.linalg import block_diag, sqrtm
import scipy
import scipy.linalg
import numpy as np
import IPython
import pdb


def construct_coeff_matrix(x_dim, B):
    # c = s*dim + Tr Z
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
    p_dim = int(1 + 2 * x_dim * (x_dim + 1) / 2)
    G = spmatrix([], [], [], (g_dim ** 2, p_dim), 'd')
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
                (it, jt) = (j, i)
            else:
                (it, jt) = (i, j)
            vec_pos = int(prev + jt * (jt + 1) / 2 + it)  # pos in params
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
    prev = int(1 + x_dim * (x_dim + 1) / 2)
    for j in range(x_dim):  # cols
        for i in range(x_dim):  # rows
            mat_pos = left * g_dim + j * g_dim + top + i
            if i >= j:
                (it, jt) = (j, i)
            else:
                (it, jt) = (i, j)
            vec_pos = int(prev + jt * (jt + 1) / 2 + it)  # pos in params
            G[mat_pos, vec_pos] += 1.
    # Block Matrix 2
    g2_dim = 2 * x_dim
    # Third Block Column
    left = 0 * x_dim + g1_dim
    top = 0 * x_dim + g1_dim
    prev = int(1 + x_dim * (x_dim + 1) / 2)
    for j in range(x_dim):  # cols
        for i in range(x_dim):  # rows
            mat_pos = left * g_dim + j * g_dim + top + i
            if i >= j:
                (it, jt) = (j, i)
            else:
                (it, jt) = (i, j)
            vec_pos = int(prev + jt * (jt + 1) / 2 + it)  # pos in params
            G[mat_pos, vec_pos] += -1.
    # Fourth Block Column
    # -------------------
    # Block Matrix 3
    g3_dim = x_dim
    # Fifth Block Column
    # Q
    left = 0 * x_dim + g1_dim + g2_dim
    top = 0 * x_dim + g1_dim + g2_dim
    prev = int(1 + x_dim * (x_dim + 1) / 2)
    for j in range(x_dim):  # cols
        for i in range(x_dim):  # rows
            mat_pos = left * g_dim + j * g_dim + top + i
            if i >= j:
                (it, jt) = (j, i)
            else:
                (it, jt) = (i, j)
            vec_pos = int(prev + jt * (jt + 1) / 2 + it)  # pos in params
            G[mat_pos, vec_pos] += 1.
    # Block Matrix 4
    g4_dim = x_dim
    # Sixth Block Column
    # Z
    left = 0 * x_dim + g1_dim + g2_dim + g3_dim
    top = 0 * x_dim + g1_dim + g2_dim + g3_dim
    prev = 1
    for j in range(x_dim):  # cols
        for i in range(x_dim):  # rows
            mat_pos = left * g_dim + j * g_dim + top + i
            if i >= j:
                (it, jt) = (j, i)
            else:
                (it, jt) = (i, j)
            vec_pos = int(prev + jt * (jt + 1) / 2 + it)  # pos in params
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
    F = real(sqrtm(B + epsilon * eye(x_dim)))
    # Construct B1
    H1 = zeros((2 * x_dim, 2 * x_dim))
    H1[:x_dim, x_dim:] = F
    H1[x_dim:, :x_dim] = F.T
    H1 = matrix(H1)

    # Construct B2
    H2 = zeros((2 * x_dim, 2 * x_dim))
    H2[:x_dim, :x_dim] = D
    H2[:x_dim, x_dim:] = A
    H2[x_dim:, :x_dim] = A.T
    Dinv = pinv(D)
    # For symmmetricity
    Dinv = (Dinv + Dinv.T) / 2.
    H2[x_dim:, x_dim:] = Dinv
    # Add this small offset in hope of correcting for the numerical
    # errors in pinv
    eps = 1e-4
    H2 += eps * eye(2 * x_dim)
    H2 = matrix(H2)

    # Construct B3
    H3 = zeros((x_dim, x_dim))
    H3 = matrix(H3)

    # Construct B4
    H4 = zeros((x_dim, x_dim))
    H4 = matrix(H4)

    # Construct Block matrix
    H = spdiag([H1, H2, H3, H4])
    hs = [H]
    return hs, F


def solve_Q(x_dim, A, B, D, max_iters, show_display):
    # x = [s vec(Z) vec(Q)]
    epsilon = np.finfo(np.float32).eps
    F = real(sqrtm(B + epsilon * eye(x_dim)))
    c_dim = int(1 + 2 * x_dim * (x_dim + 1) / 2)
    c = zeros((c_dim, 1))
    # c = s*dim + Tr Z
    c[0] = x_dim
    prev = 1
    for i in range(x_dim):
        vec_pos = int(prev + i * (i + 1) / 2 + i)
        c[vec_pos] = 1.
    cm = matrix(c)

    # Scale objective down by T for numerical stability
    eigs = eig(B)[0]
    # B may be a zero matrix (if no datapoints were associated here).
    T = max(abs(max(eigs)), abs(min(eigs)))
    if T != 0.:
        Bdown = B / T
    else:
        Bdown = B
    # Ensure that D doesn't have negative eigenvals
    # due to numerical issues
    min_D_eig = min(eig(D)[0])
    if min_D_eig < 0:
        # assume abs(min_D_eig) << 1
        D = D + 2 * abs(min_D_eig) * eye(x_dim)
    # Ensure that D - A D A.T is PSD. Otherwise, the problem is
    # unsolvable and weird numerical artifacts can occur.
    min_Q_eig = min(eig(D - dot(A, dot(D, A.T)))[0])
    if min_Q_eig < 0:
        eta = 0.99
        power = 1
        while (min(eig(D - dot((eta ** power) * A,
                               dot(D, (eta ** power) * A.T)))[0]) < 0):
            power += 1
        A = (eta ** power) * A
    Gs = construct_coeff_matrix(x_dim, Bdown)
    for i in range(len(Gs)):
        Gs[i] = -Gs[i]
    G = np.copy(matrix(Gs[0]))

    # Now scale D upwards for stability
    max_D_eig = max(eig(D)[0])

    hs, _ = construct_const_matrix(x_dim, A, Bdown, D)
    for i in range(len(hs)):
        hs[i] = matrix(hs[i])
    # Smallest number epsilon such that 1. + epsilon != 1.
    epsilon = np.finfo(np.float32).eps
    # Add a small positive offset to avoid taking sqrt of singular matrix
    F = real(sqrtm(Bdown + epsilon * eye(x_dim)))

    solvers.options['maxiters'] = max_iters
    solvers.options['show_progress'] = show_display
    #solvers.options['debug'] = True
    sol = solvers.sdp(cm, Gs=Gs, hs=hs)
    qvec = np.array(sol['x'])
    qvec = qvec[int(1 + x_dim * (x_dim + 1) / 2):]
    Q = np.zeros((x_dim, x_dim))
    for j in range(x_dim):
        for k in range(j + 1):
            vec_pos = int(j * (j + 1) / 2 + k)
            Q[j, k] = qvec[vec_pos]
            Q[k, j] = Q[j, k]
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
