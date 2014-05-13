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
from feasibility_sdp_solver import *
import scipy.optimize


class GeneralSolver(object):
    """ Implementation of a convex solver on the semidefinite cone, which
    uses binary search and the FeasibilitySolver below to solve general
    semidefinite cone convex programs.
    """
    def __init__(self, R, L, U, dim, eps):
        self.R = R
        self.L = L
        self.U = U
        self.dim = dim
        self.eps = eps
        self._feasibility_solver = FeasibilitySolver(R, dim, eps)

    def save_constraints(self, obj, grad_obj, As, bs, Cs, ds,
            Fs, gradFs, Gs, gradGs):
        (self.As, self.bs, self.Cs, self.ds,
            self.Fs, self.gradFs, self.Gs, self.gradGs) = \
                As, bs, Cs, ds, Fs, gradFs, Gs, gradGs
        self.obj = obj
        self.grad_obj = grad_obj

    def create_feasibility_solver(self, fs, grad_fs):
        As, bs, Cs, ds, Fs, gradFs, Gs, gradGs = \
            (self.As, self.bs, self.Cs, self.ds,
                self.Fs, self.gradFs, self.Gs, self.gradGs)
        newFs = Fs + fs
        newGradFs = gradFs + grad_fs
        f = FeasibilitySolver(self.R, self.dim, self.eps)
        f.init_solver(As, bs, Cs, ds, newFs, newGradFs, Gs, gradGs)
        return f

    def solve(self, N_iter, tol, X_init=None, interactive=False):
        """
        Solves optimization problem

        min h(X)
        subject to
            Tr(A_i X) <= b_i, Tr(C_j X) == d_j
            f_k(X) <= 0, g_l(X) == 0
            Tr(X) <= R
        assuming
            L <= h(X) <= U

        where h is convex.

        We perform binary search to minimize h(X). We choose alpha \in
        [U,L]. We ascertain whether alpha is a feasible value of h(X) by
        performing two subproblems:

        (1)
        Feasibility of X
        subject to
            Tr(A_i X) <= b_i, Tr(C_i X) == d_i
            f_k(X) <= 0, g_l(X) == 0
            L <= h(X) <= alpha
            Tr(X) <= R

        and

        (2)
        Feasibility of X
        subject to
            Tr(A_i X) <= b_i, Tr(C_i X) == d_i
            f_k(X) <= 0, g_l(X) == 0
            alpha <= h(X) <= U
            Tr(X) == 1

        If problem (1) is feasible, then we know that there is a solution
        in range [L, alpha]. If problem (2) is feasible, then there is a
        solution in range [alpha, U]. We then recurse.

        Parameters
        __________
        N_iter: int
            Max number of iterations for each feasibility search.
        """
        # Do the binary search
        X_L = None
        X_U = None
        succeed = False
        U, L = self.U, self.L
        # Test that problem is originally feasible
        f_lower = self.create_feasibility_solver([], [])
        _, _, succeed = f_lower.feasibility_solve(N_iter, tol,
                methods=['frank_wolfe', 'frank_wolfe_stable'],
                disp=True)
        if not succeed:
            if interactive:
                print "Problem infeasible with obj in (%f, %f)" % (L, U)
                wait = raw_input("Press ENTER to continue")
                pass
            return (None, U, X_U, L, X_L, succeed)
        # If we get here, then the problem is feasible
        if interactive:
            print "Problem feasible with obj in (%f, %f)" % (L, U)
            wait = raw_input("Press ENTER to continue")
            print
        while (U - L) >= tol:
            alpha = (U + L) / 2.0
            if interactive:
                print "Checking feasibility in (%f, %f)" % (L, alpha)
                pass
            h_alpha = lambda X: self.obj(X) - alpha
            grad_h_alpha = lambda X: self.grad_obj(X)
            f_lower = self.create_feasibility_solver([h_alpha],
                    [grad_h_alpha])
            X_L, fX_L, succeed_L = f_lower.feasibility_solve(N_iter, tol,
                    methods=['frank_wolfe', 'frank_wolfe_stable'],
                    disp=True)
            if interactive:
                print "Checked feasibility in (%f, %f)" % (L, alpha)
                pass
            if succeed_L:
                U = alpha
                if interactive:
                    print "Problem feasible with obj in (%f, %f)" % (L, U)
                    wait = raw_input("Press ENTER to continue")
                    pass
                continue
            else:
                if interactive:
                    print "Problem infeasible with obj in (%f, %f)" \
                            % (L, U)
                    wait = raw_input("Press ENTER to continue")
                    print "\tContinuing search in (%f, %f)" % (L, U)
                    L = alpha
                    pass
                continue
            break

        if (U - L) <= tol:
            succeed = True
        return (alpha, U, X_U, L, X_L, succeed)
