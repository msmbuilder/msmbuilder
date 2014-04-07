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
import pdb

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
            The curvature constant of function f (Optional).
        """
        v = random.rand(dim, 1)
        X = np.outer(v, v)
        X /= np.trace(X)
        for k in range(N_iter):
            grad = gradf(X)
            if dim >= 3:
                if Cf != None:
                    epsk = Cf/(k+1)**2
                    _, vk = linalg.eigs(grad, k=1, tol=epsk, which='LR')
                else:
                    _, vk = linalg.eigs(grad, k=1, which='LR')
            else:
                ws, vs = np.linalg.eig(grad)
                i = np.argmax(np.real(ws))
                vk = vs[:,i]

            # Avoid strange errors with complex numbers
            vk = np.real(vk)
            alphak = min(1.,2./(k+1))
            X = X + alphak * (np.outer(vk,vk) - X)
        return X

class GeneralSDPHazanSolver(object):
    """ Implementation of Hazan's Fast SDP solver, which uses
        the bounded trace PSD solver above to solve general SDPs.
    """
    def __init__(self):
        self._solver = BoundedTraceSDPHazanSolver()

    def solve(self, C, As, bs, eps, dim, R):
        """
        Solves optimization problem
        max Tr(CX)
        subject to
            Tr(A_i X) <= b_i
            Tr(X) <= R

        Solution of this problem with Frank-Wolfe methods requires
        two transformations. In the first, we normalize the trace
        upper bound Tr(X) <= R by rescaling

            A_i := R A_i

        (No need to rescale C since the max operator removes constant
        factors). After the transformation, we assume Tr(X) <= 1. Next,
        we derive an upper bound on Tr(CX). Note that

            (Tr(CX))^2 = (\sum_{i,j} C_{ij} X_{ij})^2
                       <= (\sum_{i,j} C_{ij}^2) (\sum_{i,j} X_{ij}^2)
                       <= (\sum_{i,j} C_{ij}^2) (\sum_{i,j} X_{ii} X_{jj})
                       <= (\sum_{i,j} C_{ij}^2) (\sum_{i} X_{ii})^2
                       <= (\sum_{i,j} C_{ij}^2)

        The first inequality is Cauchy-Schwarz. The second inequality
        follows from a standard fact about semidefinite matrices (CITE).
        The inequality follows again from Cauchy-Schwarz. The last
        inequality follows from the fact that Tr(X) <= 1. Similarly,

            Tr(CX) >= 0

        By the fact that X is PSD (CITE). Let D = \sum_{i,j} C_{ij}^2. We
        perform the rescaling.

            C:= (1/D) C

        After the scaling transformation, we have that 0 <= Tr(CX) <= 1.
        The next required transformation is binary search. Choose value
        alpha \in [0,1]. We ascertain whether alpha is a feasible value of
        Tr(CX) by performing two subproblems:

        (1)
        Feasibility of X
        subject to
            Tr(A_i X) <= b_i
            Tr(X) <= 1
            Tr(C X) <= alpha

        and

        (2)
        Feasibility of X
        subject to
            Tr(A_i X) <= b_i
            Tr(X) <= 1
            Tr(C X) >= alpha => Tr(-C X) <= alpha

        If problem (1) is feasible, then we know that the original problem
        has a solution in range [0, alpha]. If problem (2) is feasible,
        then the original problem has solution in range [alpha, 1]. We can
        use these indicators to perform binary search to find optimum
        alpha*. Thus, we need only solve the feasibility subproblems.

        To solve this problem, note that the the diagonal entries of
        positive semidefinite matrices are real and nonnegative.
        Consequently, we introduce variables

        Y  := [[X, 0],  F_i  := [[A_i, 0],  G = [[C, 0],
               [0, y]]           [ 0,  0]]       [0, 0]]

        Y is PSD if and only if X is PSD and y is real and nonnegative. The
        constraint that Tr Y = 1 is true if and only if Tr X <= 1. Thus,
        we can recast a feasibility problem as

        Feasibility of Y
        subject to
            Tr(F_i Y) <= b_i
            Tr(Y) = 1

        This is the format solvable by Hazan's algorithm.

        Parameters
        __________
        C: np.ndarray
            The objective function
        As: list
            A list of square (dim, dim) numpy.ndarray matrices
        bs: list
            A list of floats
        eps: float
            Allowed error tolerance. Must be > 0
        dim: int
            Dimension of input
        R: float
            Upper bound on trace of X: 0 <= Tr(X) <= R
        """
        m = len(As)
        Aprimes = []
        # Rescale the trace bound
        for i in range(m):
            Aprime = R * As[i]
            Aprimes.append(Aprime)
        As = Aprimes

        # Rescale the optimization matrix
        D = sum(C * C) # note this is a Hadamard product, not dot
        C = (1./D) * D

        # Expand all constraints to be expressed in terms of Y
        Fs = []
        for i in range(m):
            Ai = As[i]
            Fi = zeros((dim+1,dim+1))
            Fi[:dim, :dim] = Ai
            Fs.append(F)
        G = zeros((dim+1,dim+1))
        G[:dim, :dim] = C

        # Do the binary search
        upper = 1.0
        lower = 0.0
        FAIL = False
        X_LOWER = None
        X_UPPER = None
        while (upper - lower) >= eps:
            alpha = (upper + lower) / 2.0
            # Check feasibility in [lower, alpha]
            Fs.append(G)
            bs.append(alpha)
            Y_LOWER, _, FAIL_LOWER = self.feasibility_solve(Fs, es,
                                                            eps, dim)
            Fs.pop()
            bs.pop()

            # Check feasibility in [alpha, upper]
            Fs.append(-G)
            bs.append(alpha)
            Y_UPPER, _, FAIL_UPPER= self.feasibility_solve(Fs, es,
                                                           eps, dim)

            if not FAIL_UPPER:
                X_UPPER = Y_UPPER[:dim,:dim]
                lower = alpha
            elif not FAIL_LOWER:
                X_LOWER = X_LOWER
                upper = alpha
            else:
                FAIL = TRUE
                break
        alpha_star = None
        X_star = None
        if not FAIL:
            alpha_star = (upper + lower)/2.
            X_star = X_UPPER
        return (alpha_star, X_star, FAIL)


    def feasibility_solve(self, As, bs, eps, dim):
        """
        Implements the subproblem of solving feasibility problems of the
        type

        Feasibility of X
        subject to
            Tr(A_i X) <= b_i
            Tr(X) = 1

        by optimizing the function

        f(X) = -(1/M) log(sum_{i=1}^m exp(M*(A_i dot X - b_i)))

        where m is the number of linear constraints and M = log m / eps,
        with eps an error tolerance parameter

        Parameters
        __________
        As: list
            A list of square (dim, dim) numpy.ndarray matrices
        bs: list
            A list of floats
        eps: float
            Allowed error tolerance. Must be > 0
        dim: int
            Dimension of input
        """
        # Do some type checking
        # This is awkward; think of a better way to do this
        CORRECT_ARGS= True
        if isinstance(As, list) and isinstance(bs,list):
            if len(As) == len(bs):
                for i in range(len(As)):
                    Ai = As[i]
                    bi = bs[i]
                    if (np.shape(Ai) != (dim, dim) or
                            (not isinstance(bi, float))):
                        CORRECT_ARGS = False
                        break
        if not CORRECT_ARGS:
            raise ValueError(
            """Incorrect Arguments to feasibility_solve.
            As should be a list of square matrices of shape (dim,
            dim), while bs should be a list of floats. Needs
            len(As) = len(bs).
            """)

        def f(X):
            """
            X: np.ndarray
                Computes function
                f(X) = -(1/M) log(sum_{i=1}^m exp(M*(Tr(Ai,X) - bi)))
            """
            s = 0.
            for i in range(m):
                Ai = As[i]
                bi = bs[i]
                s += np.exp(M*(np.trace(np.dot(Ai,X)) - bi))
            return -(1.0/M) * np.log(s)

        def gradf(X):
            """
            X: np.ndarray
                Computes grad f(X) = -(1/M) * f' / f where
                  f' = sum_{i=1}^m exp(M*(Tr(Ai, X) - bi)) * (M * Ai.T)
                  f  = sum_{i=1}^m exp(M*(Tr(Ai,X) - bi))
            """
            num = 0.
            denom = 0.
            for i in range(m):
                Ai = As[i]
                bi = bs[i]
                if dim >= 2:
                    num += np.exp(M*(np.trace(np.dot(Ai,X)) - bi))*(M*Ai.T)
                    denom += np.exp(M*(np.trace(np.dot(Ai,X)) - bi))
                else:
                    num += np.exp(M*(Ai*X - bi))*(M*Ai.T)
                    denom += np.exp(M*(Ai*X - bi))
            return (-1.0/M) * num/denom
        m = len(As)
        M = np.max((np.log(m)/eps, 1.))
        K = int(1/eps)

        X = self._solver.solve(f, gradf, dim, K)
        fX = f(X)
        print("X:")
        print X
        print("f(X) = %f" % (fX))
        FAIL = (fX < -eps)
        print("FAIL: " + str(FAIL))
        return X, fX, FAIL

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


## Do a simple test of the Bounded Trace Solver
#dim = 4
## Note that H(-f) = 2 I (H is the hessian)
#Cf = 2.
#N_iter = 100
## Now do a dummy optimization problem. The
## problem we consider is
## max - \sum_k x_k^2
## such that \sum_k x_k = 1
## The optimal solution is -1/n, where
## n is the dimension.
#b = BoundedTraceSDPHazanSolver()
#b.solve(f, gradf, dim, N_iter, Cf=Cf)


## Do a simple test of the feasibility solver
#dim = 2
#
## Check argument validation
#ERROR = False
#try:
#    g = GeneralSDPHazanSolver()
#    As = [np.array([[1.5, 0.],
#                    [0., 1.5]])]
#    bs = [np.array([1.5, 0])]
#    eps = 1e-1
#    dim = 1
#    g.feasibility_solve(As, bs, eps, dim)
#except ValueError:
#    ERROR = True
#    pass
#assert ERROR == True
#
## Now try two-dimensional basic feasible example
#g = GeneralSDPHazanSolver()
#As = [np.array([[1, 0.],
#                [0., 2]])]
#bs = [1.5]
#eps = 1e-1
#dim = 2
#X, fX, FAIL = g.feasibility_solve(As, bs, eps, dim)
#assert FAIL == False
#
## Now try two-dimensional basic infeasibility example
#g = GeneralSDPHazanSolver()
#As = [np.array([[2, 0.],
#                [0., 2]])]
#bs = [1.]
#eps = 1e-1
#dim = 2
#X, fX, FAIL = g.feasibility_solve(As, bs, eps, dim)
#assert FAIL == True

# Do a simple test of General SDP Solver with binary search
g = GeneralSDPHazanSolver()
As = [np.array([[1., 2.],
                [1., 2.]])]
bs = [np.array([1., 1.])]
