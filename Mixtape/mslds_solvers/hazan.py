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
import scipy.linalg
import numpy.random as random
import numpy as np
import pdb
import time
from numbers import Number
from hazan_penalties import *
from hazan_utils import *
import scipy.optimize

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
                num_tries=5, alphas=None, X_init=None):
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
        # orthonormalize v
        v = v / np.linalg.norm(v)
        if X_init == None:
            X = np.outer(v, v)
        else:
            X = np.copy(X_init)
        #X /= np.trace(X)
        import pdb
        pdb.set_trace()
        for j in range(N_iter):
            grad = gradf(X)
            print "\tIteration %d" % j
            if DEBUG:
                print "\tOriginal X:\n", X
                print "\tgrad X:\n", grad
            if dim >= 3:
                if Cf != None:
                    epsj = Cf/(j+1)
                else:
                    epsj = 1e-6
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
            if alphas == None:
                #alphaj = min(1.,2./(j+1))
                alphaj = min(.5,2./(j+1))
            else:
                alphaj = alphas[j]
            O = np.outer(vj, vj)
            step = (np.outer(vj,vj) - X)
            gamma = 1.0
            # Do simple back-tracking line search
            scale_down = 0.7
            f_X = f(X)
            gamma_best = gamma
            gamma_best_proj = gamma
            f_best = f((1.-gamma)*X + gamma*step)
            X_prop = (1 - gamma_best) * X + gamma_best*step
            f_best_proj = f_X
            N_tries = 30
            for count in range(N_tries):
                if f_best < f_X and count > N_tries:
                    break
                gamma = scale_down * gamma
                f_cur = f((1.-gamma)*X + gamma*step)
                if f_best < f_cur:
                    f_best = f_cur
                    gamma_best = gamma
                    X_prop = (1.-gamma)*X + gamma*step
                #print "\t\tf_cur: ", f_cur
                X_proj = X + gamma * grad
                X_proj = scipy.linalg.sqrtm(np.dot(X_proj.T, X_proj))
                f_cur_proj = f(X_proj)
                if f_best < f_cur_proj:
                    f_best = f_cur_proj
                    X_prop = X_proj
            if DEBUG:
                print "\tf(X):\n", f(X)
                print "\talphaj:\n", alphaj
                print "\tvk vk.T:\n", np.outer(vj,vj)
                print "\tstep:\n", step
            if f(X_prop) > f(X):
                X = X_prop
            print "\t\tgamma: ", gamma_best
            print "\t\t\tf(X): ", f(X)
            #X = X + alphaj * (np.outer(vj,vj) - X)
        import pdb
        pdb.set_trace()
        return X


class FeasibilitySDPHazanSolver(object):
    """ Implementation of Hazan's Fast SDP feasibility, which uses
        the bounded trace PSD solver above to solve general SDPs.
    """
    def __init__(self):
        self._solver = BoundedTraceSDPHazanSolver()

    def feasibility_grad(self, X, As, bs, Cs, ds, Fs, gradFs, Gs,
            gradGs, eps):
        (dim, _) = np.shape(X)
        m = len(As)
        n = len(Cs)
        p = len(Fs)
        q = len(Gs)
        M = compute_scale_full(m, n, p, q, eps)
        def gradf(X):
            return neg_max_grad_penalty(X, M, As, bs, Cs, ds,
                    Fs, gradFs, Gs, gradGs, eps)
        return gradf(X)

    def feasibility_val(self, X, As, bs, Cs, ds, Fs, Gs, eps):
        (dim, _) = np.shape(X)
        m = len(As)
        n = len(Cs)
        p = len(Fs)
        q = len(Gs)
        M = compute_scale(m, n, p, q, eps)
        def f(X):
            return neg_max_penalty(X, M, As, bs, Cs, ds, Fs, Gs)
        return f(X)

    def feasibility_solve(self, As, bs, Cs, ds, Fs, gradFs, Gs, gradGs,
            eps, dim, N_iter=None, X_init=None):
        """
        Solves feasibility problems of the type

        Feasibility of X
        subject to
            Tr(A_i X) <= b_i, Tr(C_j X)  = d_j
            f_k(X) <= 0, g_l(X) == 0
            Tr(X) = 1

        by optimizing neg_max_penalty function
        TODO: Switch to log_sum_exp_penalty once numerically stable

        Parameters
        __________
        As: list
            inequality square (dim, dim) numpy.ndarray matrices
        bs: list
            inequality floats
        Cs: list
            equality square (dim, dim) numpy.ndarray matrices
        ds: list
            equality floats
        Fs: list
            convex inequalities
        Gs: list
            convex equalities
        eps: float
            Allowed error tolerance. Must be > 0
        dim: int
            Dimension of input
        """

        m = len(As)
        n = len(Cs)
        p = len(Fs)
        q = len(Gs)
        M = compute_scale_full(m, n, p, q, eps)
        if N_iter == None:
            N_iter = int(1./eps)
        # Need to swap in some robust theory about Cf
        fudge_factor = 1.0
        def f(X):
            return neg_max_penalty(X, M, As, bs, Cs, ds, Fs, Gs)
        def gradf(X):
            return log_sum_exp_grad_penalty(X, M, As, bs, Cs, ds,
                    Fs, gradFs, Gs, gradGs, eps)

        #import pdb
        #pdb.set_trace()
        start = time.clock()
        X = self._solver.solve(f, gradf, dim, N_iter, X_init=X_init)
        elapsed = (time.clock() - start)
        fX = f(X)
        print "\tX:\n", X
        print "\tf(X) = %f" % (fX)
        SUCCEED = not (fX < -fudge_factor*eps)
        print "\tSUCCEED: " + str(SUCCEED)
        print "\tComputation Time (s): ", elapsed
        #import pdb
        #pdb.set_trace()
        return X, fX, SUCCEED


class GeneralSDPHazanSolver(object):
    """ Implementation of a SDP solver, which uses binary search
        and the FeasibilitySDPSolver below to solve general SDPs.
    """
    def __init__(self):
        self._solver = FeasibilitySDPHazanSolver()

    def solve(self, h, gradh, As, bs, Cs, ds, Fs, gradFs, Gs, gradGs,
            eps, dim, R, U, L, N_iter, X_init=None):
        """
        Solves optimization problem

        min h(X)
        subject to
            Tr(A_i X) <= b_i, Tr(C_j X) == d_j
            f_k(X) <= 0, g_l(X) == 0
            Tr(X) <= R
        assuming
            L <= h(X) <= U

        where h is convex.  Solution of this problem with Frank-Wolfe
        methods requires two transformations. For the first, we normalize
        the trace upper bound Tr(X) <= R to Tr(X) <= 1 by performing
        change of variable

            X := X / R

        To keep the inequality constraints in their original format,
        we need to perform scalings

            A_i         := A_i * R
            C_i         := C_i * R
            f_k(X)      := f_k(R * X)
            grad f_k(X) := grad g_k(R * X) * R
            g_l(X)      := g_l(R * X)
            grad g_l(X) := grad g_l(R * X) * R
            h(X)        := h(R * X)
            grad h(X)   := R * grad h(R * X)

        To transform inequality Tr(X) <= 1 to Tr(X) == 1, we perform
        change of variable

        Y  := [[X,    0     ],
               [0, 1 - tr(X)]]

        Y is PSD if and only if X is PSD and y is real and nonnegative.
        The constraint that Tr Y = 1 is true if and only if Tr X <= 1.
        We transform the origin constraints by

         A_i := [[A_i, 0],  C_i := [[C_i, 0],
                 [ 0,  0]]          [ 0,  0]]

        f_k(Y)      := f_k(X)
        grad f_k(Y) := [[ grad f_k(X), 0], # Multiply this by 1/R?
                        [      0     , 0]]
        g_l(Y)      := g_l(X)
        grad g_l(Y) := [[ grad g_l(X), 0], # Multiply this by 1/R?
                        [      0     , 0]]

        h(Y)        := h(X)
        grad h(Y)   := [[ grad h(X), 0], # Multiply this by 1/R?
                        [      0   , 0]]

        Now we can constrain Tr(Y) == 1.  We assume that L
        <= h(Y) <= U.  The next required operation is binary search.
        Choose value alpha \in [U,L]. We ascertain whether alpha is a
        feasible value of h(X) by performing two subproblems:

        (1)
        Feasibility of X
        subject to
            Tr(A_i Y) <= b_i, Tr(C_i Y) == d_i
            f_k(Y) <= 0, g_l(Y) == 0
            L <= h(Y) <= alpha
            Tr(Y) == 1

        and

        (2)
        Feasibility of Y
        subject to
            Tr(A_i Y) <= b_i, Tr(C_i Y) == d_i
            f_k(Y) <= 0, g_l(Y) == 0
            alpha <= h(Y) <= U
            Tr(Y) == 1

        If problem (1) is feasible, then we know that there is a solution
        in range [L, alpha]. If problem (2) is feasible, then there is a
        solution in range [alpha, U]. We can perform binary search to find
        optimum alpha*.

        Parameters
        __________
        h: np.ndarray
            The convex objective function
        As: list
            inequality square (dim, dim) numpy.ndarray matrices
        bs: list
            inequality floats
        Cs: list
            equality squares
        ds: list
            equality floats
        Fs: list
            inequality convex functions
        gradFs: list
            gradients
        Gs: list
            equality convex functions
        gradGs: list
            equality gradients
        eps: float
            Allowed error tolerance. Must be > 0
        dim: int
            Dimension of input
        R: float
            Upper bound on trace of X: 0 <= Tr(X) <= R
        """

        m = len(As)
        n = len(Cs)
        p = len(Fs)
        q = len(Gs)

        Aprimes = []
        Cprimes = []
        Fprimes = []
        gradFprimes = []
        Gprimes = []
        gradGprimes = []

        # Rescale the trace bound and expand all constraints to be
        # expressed in terms of Y
        if X_init != None:
            Y_init = np.zeros((dim+1, dim+1))
            Y_init[:dim, :dim] = X_init
            Y_init = Y_init / R
            init_trace = np.trace(Y_init)
            Y_init[dim, dim] = 1 - init_trace
        else:
            Y_init = None
        for i in range(m):
            A = R * As[i]
            Aprime = np.zeros((dim+1,dim+1))
            Aprime[:dim, :dim] = A
            Aprimes.append(Aprime)
        for j in range(n):
            C = R * Cs[j]
            Cprime = np.zeros((dim+1,dim+1))
            Cprime[:dim, :dim] = C
            Cprimes.append(Cprime)
        for k in range(p):
            fk = Fs[k]
            gradfk = gradFs[k]
            def make_fprime(fk):
                return lambda Y: fk(R * Y[:dim,:dim])
            fprime = make_fprime(fk)
            Fprimes.append(fprime)
            def make_gradfprime(gradfk):
                def gradfprime(Y):
                    ret_grad = np.zeros((dim+1,dim+1))
                    ret_grad[:dim,:dim] = gradfk(R * Y[:dim,:dim])
                    ret_grad = (1./R) * ret_grad #?
                    return ret_grad
                return gradgfprime
            gradFprimes.append(gradfprime)
        for l in range(q):
            gl = Gs[l]
            gradgl = gradGs[l]
            def make_gprime(gl):
                return lambda Y: gl(R * Y[:dim,:dim])
            gprime = make_gprime(gl)
            Gprimes.append(gprime)
            def make_gradgprime(gradgl):
                def gradgprime(Y):
                    ret_grad = np.zeros((dim+1,dim+1))
                    ret_grad[:dim, :dim] = gradgl(R * Y[:dim,:dim])
                    ret_grad = (1./R) * ret_grad #?
                    return ret_grad
                return gradgprime
            gradgprime = make_gradgprime(gradgl)
            gradGprimes.append(gradgprime)

        hprime = lambda Y: h(R * Y[:dim, :dim])
        def gradhprime(Y):
            ret_grad = np.zeros((dim+1,dim+1))
            ret_grad[:dim, :dim] = gradh(R * Y[:dim, :dim])
            ret_grad = (1./R) * ret_grad #?
            return ret_grad

        bprimes = bs
        dprimes = ds

        # Do the binary search
        SUCCEED = False
        X_L = None
        X_U = None
        while (U - L) >= eps:
            print
            print("upper: %f" % U)
            print("lower: %f" % L)
            alpha = (U + L) / 2.0
            print "Checking feasibility in (%f, %f)" % (L, alpha)
            print
            h_alpha = lambda Y: hprime(Y) - alpha
            grad_h_alpha = lambda Y: gradhprime(Y)
            h_lower = lambda Y: -hprime(Y) + L
            grad_h_lower = lambda Y: -gradhprime(Y)

            Fprimes += [h_lower, h_alpha]
            gradFprimes += [grad_h_lower, grad_h_alpha]
            import pdb
            pdb.set_trace()
            Y_L, fY_L, SUCCEED_L = self._solver.feasibility_solve(Aprimes,
                    bprimes, Cprimes, dprimes, Fprimes, gradFprimes,
                    Gprimes, gradGprimes, eps, dim+1, N_iter, Y_init)
            Fprimes = Fprimes[:-2]
            gradFprimes = gradFprimes[:-2]
            print "Checked feasibility in (%f, %f)" % (L, alpha)
            import pdb
            pdb.set_trace()
            if SUCCEED_L:
                X_L = R * Y_L[:dim, :dim]
                U = alpha
                continue

            # Check feasibility in [alpha, U]
            print
            print "Checking feasibility in (%f, %f)" % (alpha, U)
            print "alpha: ", -alpha
            h_alpha = lambda Y: -hprime(Y) + alpha
            grad_h_alpha = lambda(Y): -gradhprime(Y)
            h_upper = lambda Y: hprime(Y) - U
            grad_h_upper = lambda Y: gradhprime(Y)
            Fprimes += [h_alpha, h_upper]
            gradFprimes += [grad_h_alpha, grad_h_upper]
            Y_U, fY_U, SUCCEED_U = self._solver.feasibility_solve(Aprimes,
                    bprimes, Cprimes, dprimes, Fprimes, gradFprimes,
                    Gprimes, gradGprimes, eps, dim+1, N_iter, Y_init)
            Fprimes = Fprimes[:-2]
            gradFprimes = gradFprimes[:-2]
            print "Checked feasibility in (%f, %f)" % (alpha, U)
            import pdb
            pdb.set_trace()
            if SUCCEED_U:
                X_U = R * Y_U[:dim,:dim]
                L = alpha
                continue

            if fY_U >= fY_L:
                X_U = R * Y_U[:dim,:dim]
                L = alpha
            else:
                X_L = R * Y_L[:dim, :dim]
                U = alpha
        if (U - L) <= eps:
            fY = fY_L
            if fY_L >= -eps:
                SUCCEED = True
        return (U, L, X_U, X_L, SUCCEED)
