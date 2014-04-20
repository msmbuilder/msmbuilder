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
    the optimization problem
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
                    #print ("\tSmall eigenvalues leading to no " +
                    #         "convergence.  Shifting upwards.")
                    shift = 1
                vj = None
                last = 0.
                for i in range(num_tries):
                    try:
                        _, vj = linalg.eigsh(grad
                                + (i+1)*shift*np.eye(dim),
                                k=1, tol=epsj, which='LA')
                    except (linalg.ArpackError,
                            linalg.ArpackNoConvergence):
                        continue
                    last = i
                    if not np.isnan(np.min(vj)):
                        break
                if vj == None or np.isnan(np.min(vj)):
                    # The gradient is singular. In this case resort
                    # to the more expensive, but more stable eigh method,
                    # which is based on a divide and conquer approach
                    # instead of Lanczos
                    print("Iteration %d: Gradient is singular" % j)
                    # Going to try next smallest singular value
                    print "Looking for largest nonzero eigenvalue"
                    vj = None
                    for k in range(2,dim):
                        try:
                            ws, vs = linalg.eigsh(grad
                                    + (i+1)*shift*np.eye(dim),
                                    k=k, tol=epsj, which='LA')
                        except (linalg.ArpackError,
                                linalg.ArpackNoConvergence):
                            continue
                        if not np.isnan(np.min(vs[:,k-1])):
                            vj = vs[:,k-1]
                            print "Picked %d-th eigenvalue" % k
                            break
                    #import pdb
                    #pdb.set_trace()
                    if vj == None:
                    #import pdb
                    #pdb.set_trace()
                    # Attempt to find fis
                        print "switching to divide and conquer"
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


class FeasibilitySDPHazanSolver(object):
    """ Implementation of Hazan's Fast SDP feasibility, which uses
        the bounded trace PSD solver above to solve general SDPs.
    """
    def __init__(self):
        self._solver = BoundedTraceSDPHazanSolver()

    def feasibility_grad(self, X, As, bs, Cs, ds, eps, dim):
        m = len(As)
        n = len(Cs)
        M = compute_scale(m, n, eps)
        def gradf(X):
            return neg_max_grad_penalty(X, m, n, M,
                        As, bs, Cs, ds, dim, eps)
        return gradf(X)

    def feasibility_val(self, X, As, bs, Cs, ds, eps, dim):
        m = len(As)
        n = len(Cs)
        M = compute_scale(m, n, eps)
        def f(X):
            return neg_max_penalty(X, m, n, M,
                        As, bs, Cs, ds, dim)
        return f(X)

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
        # Need to swap in some robust theory about Cf
        fudge_factor = 3.0
        def f(X):
            return neg_max_penalty(X, m, n, M, As, bs, Cs, ds, dim)
            #return log_sum_exp_penalty(X, m, n, M, As, bs, Cs, ds, dim)
        def gradf(X):
            #return neg_max_grad_penalty(X, m, n, M,
            #            As, bs, Cs, ds, dim,eps)
            return log_sum_exp_grad_penalty(X, m, n, M,
                        As, bs, Cs, ds, dim,eps)

        start = time.clock()
        X = self._solver.solve(f, gradf, dim, N_iter)
        elapsed = (time.clock() - start)
        fX = f(X)
        print "\tX:\n", X
        print "\tf(X) = %f" % (fX)
        SUCCEED = not (fX < -fudge_factor*eps)
        print "\tSUCCEED: " + str(SUCCEED)
        print "\tComputation Time (s): ", elapsed
        #pdb.set_trace()
        return X, fX, SUCCEED


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
            Tr(C_i X) == d_i
            Tr(X) <= R

        Solution of this problem with Frank-Wolfe methods requires
        two transformations. In the first, we normalize the trace
        upper bound Tr(X) <= R by performing change of variable

            X := X / R

        To keep the inequality constraints in their original format,
        we need to perform scalings

            A_i := A_i * R
            C_i := C_i * R

        (No need to rescale E since the max operator removes constant
        factors). After the transformation, we assume Tr(X) <= 1. Next,
        we derive an upper bound on Tr(EX). Note that

            (Tr(EX))^2 == (sum_ij E_ij X_ij)^2
                       <= (sum_ij (E_ij)^2) (sum_ij (X_ij)^2)
                       <= (sum_ij (E_ij)^2) (sum_ij X_ii X_jj)
                       == (sum_ij (E_ij)^2) (sum_i X_{ii})^2
                       == (sum_ij (E_ij)^2) Tr(X)^2
                       <= (sum_ij (E_ij)^2)

        The first inequality is Cauchy-Schwarz. The second inequality
        follows from a standard fact about semidefinite matrices (CITE).

        For PSD matrix M, |m_ij| <= sqrt(m_ii m_jj) [See Wikipedia]

        The third equality follows from factorization. The last
        inequality follows from the fact that Tr(X) <= 1. Similarly,

            Tr(EX) >= 0

        By the fact that X is PSD (CITE). Let D = sum_ij (E_ij)^2. We
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
            Tr(C_i X) == d_i
            Tr(X) <= 1
            Tr(E X) <= alpha

        and

        (2)
        Feasibility of X
        subject to
            Tr(A_i X) <= b_i
            Tr(C_i X) == d_i
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

        Y  := [[X, 0], F_i := [[A_i, 0], G = [[E, 0], C_i := [[C_i, 0],
               [0, y]]         [ 0,  0]]      [0, 0]]         [ 0,  0]]

        Y is PSD if and only if X is PSD and y is real and nonnegative.
        The constraint that Tr Y = 1 is true if and only if Tr X <= 1.
        Thus, we can recast a feasibility problem as

        Feasibility of Y
        subject to
            Tr(F_i Y) <= b_i
            Tr(H_i Y) == d_i
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
            assert len(Cs) == len(ds)
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
        n = len(Cs)
        Aprimes = []
        Cprimes = []
        # Rescale the trace bound
        for i in range(m):
            Aprime = R * As[i]
            Aprimes.append(Aprime)
            print
            print "Scale As[%d] from\n" % i
            print As[i]
            print "to\n"
            print Aprime
        for j in range(n):
            Cprime = R * Cs[j]
            Cprimes.append(Cprime)
            print
            print "Scale Cs[%d] from\n" % j
            print Cs[j]
            print "to\n"
            print Cprime
        As = Aprimes
        Cs = Cprimes

        # Rescale the optimization matrix
        D = sum(E * E) # note this is a Hadamard product, not dot
        E = (1./D) * E
        print
        print "Rescaled optimization criterion E:\n", E

        # Expand all constraints to be expressed in terms of Y
        Fs = [] # expanded As
        Hs = [] # expanded Cs
        for i in range(m):
            Ai = As[i]
            Fi = np.zeros((dim+1,dim+1))
            Fi[:dim, :dim] = Ai
            Fs.append(Fi)
            print
            print "Expand As[%d] from\n" % i
            print As[i]
            print "to\n"
            print Fi
            print "recall bs[%d] = %f" % (i, bs[i])
        for j in range(n):
            Cj = Cs[j]
            Hj = np.zeros((dim+1,dim+1))
            Hj[:dim, :dim] = Cj
            Hs.append(Hj)
            print
            print "Expand Cs[%d] from\n" % j
            print Cs[j]
            print "to\n"
            print Hj
            print "recall ds[%d] = %f" % (j, ds[j])
        G = np.zeros((dim+1,dim+1))
        G[:dim, :dim] = E
        #import pdb
        #pdb.set_trace()

        # Generate constraints required to make Y be a block matrix
        for i in range(dim):
            Ri = np.zeros((dim+1,dim+1))
            Ri[i, dim] = 1.
            ri = 0.
            Hs.append(Ri)
            ds.append(ri)
            print
            print "Adding equality constraint Tr(R%i) = 0., where " % i
            print "R%d equals\n" % i
            print Ri

            Si = np.zeros((dim+1,dim+1))
            Si[dim, i] = 1.
            si = 0.
            print
            print "Adding equality constraint Tr(S%i) = 0., where\n" % i
            print "S%d equals\n" % i
            print Si
            Hs.append(Si)
            ds.append(si)

        # Do the binary search
        upper = 1.0
        lower = 0.0
        SUCCEED = False
        X_LOWER = None
        X_UPPER = None
        while (upper - lower) >= eps:
            print
            print("upper: %f" % upper)
            print("lower: %f" % lower)
            alpha = (upper + lower) / 2.0
            # Check feasibility in [lower, alpha]
            print "Checking feasibility in (%f, %f)" % (lower, alpha)
            print "Adding inequality constraint Tr(GX) <= alpha"
            print "G:\n", G
            print "alpha: ", alpha
            print
            Fs.append(G)
            bs.append(alpha)
            Y_LOWER, _, SUCCEED_LOWER = self._solver.feasibility_solve(Fs,
                    bs, Hs, ds, eps, dim+1)
            Fs.pop()
            bs.pop()

            # Check feasibility in [alpha, upper]
            print
            print "Checking feasibility in (%f, %f)" % (alpha, upper)
            print "Adding inequality constraint Tr(-GX) <= alpha"
            print "-G:\n", -G
            print "alpha: ", -alpha
            Fs.append(-G)
            bs.append(-alpha)
            Y_UPPER, _, SUCCEED_UPPER= self._solver.feasibility_solve(Fs,
                    bs, Hs, ds, eps, dim+1)
            Fs.pop()
            bs.pop()

            #import pdb
            #pdb.set_trace()
            if SUCCEED_UPPER:
                X_UPPER = Y_UPPER[:dim,:dim]
                lower = alpha
            elif SUCCEED_LOWER:
                X_LOWER = X_LOWER
                upper = alpha
            else:
                break
        if (upper - lower) <= eps:
            SUCCEED = True
        if X_UPPER != None:
            X_UPPER = R * X_UPPER
        if X_LOWER != None:
            X_LOWER = R * X_LOWER
        return (upper, lower, X_UPPER, X_LOWER, SUCCEED)
