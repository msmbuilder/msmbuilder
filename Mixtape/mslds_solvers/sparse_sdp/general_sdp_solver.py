"""
Implementation of Hazan's algorithm

Hazan, Elad. "Sparse Approximate Solutions to
Semidefinite Programs." LATIN 2008: Theoretical Informatics.
Springer Berlin Heidelberg, 2008, 306:316.

for approximate solution of sparse semidefinite programs.
@author: Bharath Ramsundar
@email: bharath.ramsundar@gmail.com
"""
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


class GeneralSDPSolver(object):
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
        methods requires two transformations.
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
