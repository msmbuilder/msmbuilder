"""Implementation of Hazan's algorithm

Hazan, Elad. "Sparse Approximate Solutions to
Semidefinite Programs." LATIN 2008: Theoretical Informatics.
Springer Berlin Heidelberg, 2008, 306:316.

for approximate solution of sparse semidefinite programs.
@author: Bharath Ramsundar
@email: bharath.ramsundar@gmail.com
"""
# Author: Bharath Ramsundar <bharath.ramsundar@gmail.com>
# Contributors:
# Copyright (c) 2014, Stanford University
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are
# met:
#
#   Redistributions of source code must retain the above copyright notice,
#   this list of conditions and the following disclaimer.
#
#   Redistributions in binary form must reproduce the above copyright
#   notice, this list of conditions and the following disclaimer in the
#   documentation and/or other materials provided with the distribution.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS
# IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED
# TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A
# PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
# HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
# SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED
# TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
# PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
# LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
# NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
# SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

#-----------------------------------------------------------------------------
# Imports
#-----------------------------------------------------------------------------
import scipy
import scipy.sparse.linalg as linalg
import numpy.random as random
import numpy as np

class BoundedTraceSDPHazanSolver(object):
    """ Implementation of Hazan's Algorithm, which solves
        the optimization problem:
             max f(X)
             X \in P
        where P = {X is PSD and Tr X = 1} is the set of PSD
        matrices with unit trace.
    """
    def __init__(self):
        pass
    def solve(self, f, gradf, dim, N_iter, Cf=None):
        """
        Parameters
        __________
        f: concave function
            Accepts (dim,dim) shaped matrices and outputs real
        gradf: function
            Computes grad f at given input matrix
        dim: int
            The dimensionality of the input vector space for f,
        N_iter: int
            The desired number of iterations
        Cf: float
            The curvature constant of function f
        """
        v = random.rand(dim, 1)
        X = np.outer(v, v)
        X /= np.trace(X)
        print("v:")
        print(v)
        print("X:")
        print(X)
        print("f(X) = %f" % (f(X)))
        print("rank(X) = %f" % (np.linalg.matrix_rank(X)))
        for k in range(N_iter):
            grad = gradf(X)
            print("grad_%d:" % k)
            print grad
            # Is there a way to integrate epsk into the lanczos call?
            # Ans: do tol = epsk
            if Cf != None:
                epsk = Cf/(k+1)**2
                _, vk = linalg.eigs(grad, k=1, tol=epsk, which='LR')
            else:
                _, vk = linalg.eigs(grad, k=1, which='LR')
            # Avoid strange errors with complex numbers
            vk = np.real(vk)
            alphak = min(1.,2./(k+1))
            print("alpha_%d = %f" % (k, alphak))
            X = X + alphak * (np.outer(vk,vk) - X)
            print("v_%d:" % k)
            print(vk)
            print("v_%d(v_%d).T" % (k,k))
            print(np.outer(vk,vk))
            print("X_%d:" % k)
            print(X)
            print("rank(X_%d) = %f" % (k, np.linalg.matrix_rank(X)))
            print("f(X_%d) = %f" % (k, f(X)))
        return X

class GeneralSDPHazanSolver(object):
    """ Implementation of Hazan's Fast SDP problem, which uses
        the bounded trace PSD solver above to solve general SDP's
        by optimizing the function

        f(X) = -(1/M) log(sum_{i=1}^m exp(M*(A_i dot X - b_i)))

        where m is the number of linear constraints and M = log m / eps,
        with eps an error tolerance parameter
    """
    def __init__(self):
        self._solver = BoundedTraceSDPHazanSolver()

    def solve(self, As, bs, eps, dim, N_iter):
        """
        Parameters
        __________
        As: list
            A list of square (dim, dim) numpy.ndarray matrices
        bs: list
            A list of floats
        eps: float
            Allowed error tolerance. Must be > 0
        """
        m = len(As)
        M = np.log(m)/eps
        K = int(1/eps)
        def f(X):
            """
            X: np.ndarray
                Computes function f(X) = -(1/M) log(sum_{i=1}^m exp(M*(Tr(Ai,X) - bi)))
            """
            s = 0.
            for i in range(m):
                Ai = As[i]
                bi = bs[i]
                s += np.exp(M*(np.trace(np.dot(Ai,X) - bi)))
            return -(1.0/M) * log(s)
        def gradf(X):
            """
            X: np.ndarray
                Computes grad f(X) = -(1/M) * f' / f
                where
                      f' = sum_{i=1}^m exp(M*(Tr(Ai, X) - bi)) * (M * Ai.T)
                      f  = sum_{i=1}^m exp(M*(Tr(Ai,X) - bi))
            """
            num = 0.
            denom = 0.
            for i in range(M):
                Ai = As[i]
                bi = bs[i]
                num += np.exp(M*(np.trace(Ai,X) - bi))*(M*Ai.T)
                denom += np.exp(M*(np.trace(Ai,X) - bi))
            return (-1.0/M) * num/denom
        X = None
        for k in range(K):
            X = self._solver.solve(f, gradf, dim, N_iter)
        return X

def f(x):
    """
    Computes f(x) = -\sum_k x_kk^2

    Parameters
    __________
    x: numpy.ndarray
    """
    (N, _) = np.shape(x)
    retval = 0.
    for i in range(N):
        retval += -x[i,i]**2
    return retval

def gradf(x):
    (N, _) = np.shape(x)
    G = np.zeros((N,N))
    for i in range(N):
        G[i,i] += -2.*x[i,i]
    return G

# Note that H(-f) = 2 I (H is the hessian)
Cf = 2.

#dim = 4
#N_iter = 100
## Now do a dummy optimization problem. The
## problem we consider is
## max - \sum_k x_k^2
## such that \sum_k x_k = 1
## The optimal solution is -1/n, where
## n is the dimension.
#b = BoundedTraceHazanSolver()
#b.solve(f, gradf, dim, N_iter, Cf=Cf)

dim = 1
N_iter = 100
