"""
Implementation of Hazan's algorithm

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
import time
from numbers import Number
from hazan_penalties import *
from hazan_utils import *

class BoundedTraceSDPHazanSolver(object):
    """
    Implementation of Hazan's Algorithm, which solves
    the optimization problem:
         max f(X)
         X \in P
    where P = {X is PSD and Tr X = 1} is the set of PSD
    matrices with unit trace.
    NOTE: Should probably move f, gradf into constructor
    """
    def __init__(self):
        pass
    def solve(self, f, gradf, dim, N_iter, Cf=None, DEBUG=False,
                num_tries=5):
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
        for j in range(N_iter):
            grad = gradf(X)
            print "\tIteration %d" % j
            if DEBUG:
                print "\tOriginal X:\n", X
                print "\tgrad X:\n", grad
            if dim >= 3:
                if Cf != None:
                    epsj = Cf/(j+1)**2
                else:
                    epsj = 0
                # We usually try the following eigenvector finder,
                # which is based off an Implicitly Restarted
                # Arnoldi Method (essentially a stable version of
                # Lanczos's algorithm)

                #_, vj = linalg.eigsh(grad, k=1, tol=epsj, sigma=0.,
                #        which='LM') # Gives errors for positive eigs
                # TODO: Make this more robust

                try:
                    # shift matrices upwards by a positive quantity to
                    # avoid common issues with small eigenvalues
                    w, _ = linalg.eigsh(grad, k=1, tol=epsj, which='LM')
                    if np.isnan(w) or w == -np.inf or w == np.inf:
                        shift = 1
                    else:
                        shift = 1.5*np.abs(w)
                except (linalg.ArpackError, linalg.ArpackNoConvergence):
                    shift = 1
                vj = None
                for i in range(num_tries):
                    try:
                        _, vj = linalg.eigsh(grad
                                + (i+1)*shift*np.eye(dim),
                                k=1, tol=epsj, which='LA')
                    except (linalg.ArpackError,
                            linalg.ArpackNoConvergence):
                        continue
                    if not np.isnan(np.min(vj)):
                        break
                if vj == None or np.isnan(np.min(vj)):
                    # The gradient is singular. In this case resort
                    # to the more expensive, but more stable eigh method,
                    # which is based on a divide and conquer approach
                    # instead of Lanczos
                    print("Iteration %d Switching to divide and conquer"
                            % j)
                    ws, vs = np.linalg.eigh(grad)
                    i = np.argmax(np.real(ws))
                    vj = vs[:, i]
            else:
                ws, vs = np.linalg.eig(grad)
                i = np.argmax(np.real(ws))
                vj = vs[:,i]

            # Avoid strange errors with complex numbers
            vj = np.real(vj)
            alphaj = min(1.,2./(j+1))
            step = alphaj * (np.outer(vj,vj) - X)
            if DEBUG:
                print "\talphaj:\n", alphaj
                print "\tvk vk.T:\n", np.outer(vj,vj)
                print "\tstep:\n", step
            X = X + alphaj * (np.outer(vj,vj) - X)
        return X

class GeneralSDPHazanSolver(object):
    """ Implementation of a SDP solver, which uses binary search
        and the FeasibilitySDPSolver below to solve general SDPs.
    """
    def __init__(self):
        self._solver = FeasibilitySDPHazanSolver()

    def solve(self, E, As, bs, Cs, ds, eps, dim, R):
        """
        Solves optimization problem
        max Tr(EX)
        subject to
            Tr(A_i X) <= b_i
            Tr(X) <= R

        Solution of this problem with Frank-Wolfe methods requires
        two transformations. In the first, we normalize the trace
        upper bound Tr(X) <= R by rescaling

            A_i := R A_i

        (No need to rescale E since the max operator removes constant
        factors). After the transformation, we assume Tr(X) <= 1. Next,
        we derive an upper bound on Tr(EX). Note that

            (Tr(EX))^2 = (\sum_{i,j} E_{ij} X_{ij})^2
                       <= (\sum_{i,j} E_{ij}^2) (\sum_{i,j} X_{ij}^2)
                       <= (\sum_{i,j} E_{ij}^2) (\sum_{i,j} X_{ii} X_{jj})
                       <= (\sum_{i,j} E_{ij}^2) (\sum_{i} X_{ii})^2
                       <= (\sum_{i,j} E_{ij}^2)

        The first inequality is Cauchy-Schwarz. The second inequality
        follows from a standard fact about semidefinite matrices (CITE).
        The inequality follows again from Cauchy-Schwarz. The last
        inequality follows from the fact that Tr(X) <= 1. Similarly,

            Tr(EX) >= 0

        By the fact that X is PSD (CITE). Let D = \sum_{i,j} E_{ij}^2. We
        perform the rescaling.

            E:= (1/D) E

        After the scaling transformation, we have that 0 <= Tr(EX) <= 1.
        The next required transformation is binary search. Choose value
        alpha \in [0,1]. We ascertain whether alpha is a feasible value of
        Tr(EX) by performing two subproblems:

        (1)
        Feasibility of X
        subject to
            Tr(A_i X) <= b_i
            Tr(X) <= 1
            Tr(E X) <= alpha

        and

        (2)
        Feasibility of X
        subject to
            Tr(A_i X) <= b_i
            Tr(X) <= 1
            Tr(E X) >= alpha => Tr(-E X) <= alpha

        If problem (1) is feasible, then we know that the original problem
        has a solution in range [0, alpha]. If problem (2) is feasible,
        then the original problem has solution in range [alpha, 1]. We can
        use these indicators to perform binary search to find optimum
        alpha*. Thus, we need only solve the feasibility subproblems.

        To solve this problem, note that the the diagonal entries of
        positive semidefinite matrices are real and nonnegative.
        Consequently, we introduce variables

        Y  := [[X, 0],  F_i  := [[A_i, 0],  G = [[E, 0],
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
        E: np.ndarray
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
        # Do some type checking
        try:
            assert_list_of_types(bs, Number)
            assert_list_of_square_arrays(As, dim)
            assert len(As) == len(bs)
            assert isinstance(E, np.ndarray)
            assert np.shape(E) == (dim, dim)
            assert isinstance(R, Number)
            assert R > 0
        except AssertionError:
            raise ValueError(
            """Incorrect Arguments to solve.
            As should be a list of square arrays of shape (dim,
            dim), while bs should be a list of floats. Needs
            len(As) = len(bs). E should be square array of shape
            (dim, dim) as well. R should be a real number greater than 0.
            """)


        m = len(As)
        Aprimes = []
        # Rescale the trace bound
        for i in range(m):
            Aprime = R * As[i]
            Aprimes.append(Aprime)
        As = Aprimes

        # Rescale the optimization matrix
        D = sum(E * E) # note this is a Hadamard product, not dot
        E = (1./D) * D

        # Expand all constraints to be expressed in terms of Y
        Fs = []
        for i in range(m):
            Ai = As[i]
            Fi = np.zeros((dim+1,dim+1))
            Fi[:dim, :dim] = Ai
            Fs.append(Fi)
        G = np.zeros((dim+1,dim+1))
        G[:dim, :dim] = E

        # Generate constraints required to make Y be a block matrix
        for i in range(dim):
            Ri = np.zeros((dim+1,dim+1))
            Ri[i, dim] = 1.
            ri = 0.
            Cs.append(Ri)
            ds.append(ri)

            Si = np.zeros((dim+1,dim+1))
            Si[dim, i] = 1.
            si = 0.
            Cs.append(Si)
            ds.append(si)

        # Do the binary search
        upper = 1.0
        lower = 0.0
        FAIL = False
        X_LOWER = None
        X_UPPER = None
        while (upper - lower) >= eps:
            print
            print("upper: %f" % upper)
            print("lower: %f" % lower)
            alpha = (upper + lower) / 2.0
            # Check feasibility in [lower, alpha]
            Fs.append(G)
            bs.append(alpha)
            Y_LOWER, _, FAIL_LOWER = self._solver.feasibility_solve(Fs, bs,
                    Cs, ds, eps, dim+1)
            Fs.pop()
            bs.pop()

            # Check feasibility in [alpha, upper]
            Fs.append(-G)
            bs.append(alpha)
            Y_UPPER, _, FAIL_UPPER= self._solver.feasibility_solve(Fs, bs,
                    Cs, ds, eps, dim+1)

            if not FAIL_UPPER:
                X_UPPER = Y_UPPER[:dim,:dim]
                lower = alpha
            elif not FAIL_LOWER:
                X_LOWER = X_LOWER
                upper = alpha
            else:
                FAIL = TRUE
                break
        if X_UPPER != None:
            X_UPPER = R * X_UPPER
        if X_LOWER != None:
            X_LOWER = R * X_LOWER
        return (upper, lower, X_UPPER, X_LOWER, FAIL)


class FeasibilitySDPHazanSolver(object):
    """ Implementation of Hazan's Fast SDP feasibility, which uses
        the bounded trace PSD solver above to solve general SDPs.
    """
    def __init__(self):
        self._solver = BoundedTraceSDPHazanSolver()

    def feasibility_solve(self, As, bs, Cs, ds, eps, dim):
        """
        Solves feasibility problems of the type

        Feasibility of X
        subject to
            Tr(A_i X) <= b_i
            Tr(C_i X)  = d_i
            Tr(X) = 1

        by optimizing neg_max_penalty function
        TODO: Switch to log_sum_exp_penalty once numerically stable

        Parameters
        __________
        As: list
            A list of square (dim, dim) numpy.ndarray matrices
        bs: list
            A list of floats
        Cs: list
            A list of square (dim, dim) numpy.ndarray matrices
        ds: list
            A list of floats
        eps: float
            Allowed error tolerance. Must be > 0
        dim: int
            Dimension of input
        """
        try:
            assert_list_of_types(bs, Number)
            assert_list_of_types(ds, Number)
            assert_list_of_square_arrays(As, dim)
            assert_list_of_square_arrays(Cs, dim)
            assert len(As) == len(bs)
            assert len(Cs) == len(ds)
        except AssertionError:
            raise ValueError(
            """
            Incorrect Arguments to feasibility_solve.  As, Cs should be
            lists of square matrices of shape (dim, dim), while bs, ds
            should be lists of floats. Needs len(As) == len(bs) and
            len(Cs) == len(ds).
            """)

        m = len(As)
        n = len(Cs)
        M = compute_scale(m, n, eps)
        N_iter = int(1./eps)
        #TODO: Switch to log_sum_exp_penalty once numerically stable
        def f(X):
            return neg_max_penalty(X, m, n, M, As, bs, Cs, ds, dim)
        def gradf(X):
            return neg_max_grad_penalty(X, m, n, M,
                        As, bs, Cs, ds, dim,eps)

        start = time.clock()
        X = self._solver.solve(f, gradf, dim, N_iter)
        elapsed = (time.clock() - start)
        fX = f(X)
        print "\tX:\n", X
        print "\tf(X) = %f" % (fX)
        SUCCEED = not (fX < -eps)
        print "\tSUCCEED: " + str(SUCCEED)
        print "\tComputation Time (s): ", elapsed
        #pdb.set_trace()
        return X, fX, SUCCEED
