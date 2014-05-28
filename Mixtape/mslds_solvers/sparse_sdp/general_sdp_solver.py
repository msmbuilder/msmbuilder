"""
Extended Implementation of Hazan's algorithmfor approximate solution of
sparse semidefinite programs.

Hazan, Elad. "Sparse Approximate Solutions to
Semidefinite Programs." LATIN 2008: Theoretical Informatics.
Springer Berlin Heidelberg, 2008, 306:316.


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

class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'

class GeneralSolver(object):
    """ Implementation of a convex solver on the semidefinite cone, which
    uses binary search and the FeasibilitySolver below to solve general
    semidefinite cone convex programs.
    """
    def __init__(self):
        pass

    def save_constraints(self, dim, obj, grad_obj, As, bs, Cs, ds,
            Fs, gradFs, Gs, gradGs):
        self.dim = dim
        (self.As, self.bs, self.Cs, self.ds,
            self.Fs, self.gradFs, self.Gs, self.gradGs) = \
                As, bs, Cs, ds, Fs, gradFs, Gs, gradGs
        self.obj = lambda X: obj(X)
        self.grad_obj = grad_obj

    def create_feasibility_solver(self, fs, grad_fs, eps=1e-4):
        As, bs, Cs, ds, Fs, gradFs, Gs, gradGs = \
            (self.As, self.bs, self.Cs, self.ds,
                self.Fs, self.gradFs, self.Gs, self.gradGs)
        newFs = Fs + fs
        newGradFs = gradFs + grad_fs
        f = FeasibilitySolver(self.dim, eps, As, bs, Cs, ds, newFs,
                newGradFs, Gs, gradGs)
        return f

    def print_banner(self):
        display_string = """
        #################################################
                      CONVEX SOLVER STARTED
        #################################################
        """
        display_string = bcolors.HEADER + display_string + bcolors.ENDC
        print display_string

    def interactive_wait(self, interactive):
        if interactive:
            wait = raw_input("Press ENTER to continue")

    def print_status(self, disp, debug, status, X, L, U):
        if disp:
            print "\t%s in (%f, %f)" % (status, L, U)
            if debug:
                print "\tobj(X): ", self.obj(X)
                print "\tX:\n", X

    def solve(self, N_iter, tol, search_tol, eps=1e-4, X_init=None,
        interactive=False, disp=True, verbose=False, debug=False,
        Rs = [10, 100, 1000], min_step_size=1e-6, 
        methods=['frank_wolfe', 'frank_wolfe_stable']):
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
        X = None
        succeed = False
        if disp:
            self.print_banner()
        # Test that problem is originally feasible
        f_init = self.create_feasibility_solver([], [], eps)
        X_orig, fX_orig, succeed = f_init.feasibility_solve(N_iter, tol,
                methods=methods, disp=disp,
                verbose=verbose, debug=debug, X_init = X_init, Rs=Rs,
                min_step_size=min_step_size)
        if not succeed:
            self.print_status(disp, debug, "Problem infeasible", X_orig,
                    -np.inf, np.inf)
            return (np.inf, X_orig, succeed)
        X = X_orig
        U = self.obj(X)
        if not np.isfinite(U):
            raise ValueError("Objective is not finite in GeneralSolver.solve")
        step = search_tol
        self.print_status(disp, debug, "Problem feasible", X,
                -np.inf, U)
        self.interactive_wait(interactive)
        f = f_init.get_feasibility(self.As, self.bs, self.Cs, self.ds,
                    self.Fs, self.Gs, 1e-4)
        num_stable=5
        while step >= search_tol:
            alpha = U - step 
            h_alpha = lambda X: (self.obj(X) - alpha)
            grad_h_alpha = lambda X: (self.grad_obj(X))
            f_lower = self.create_feasibility_solver([h_alpha],
                    [grad_h_alpha], eps)
            X_L, fX_L, succeed_L = f_lower.feasibility_solve(N_iter, tol,
                methods=methods, disp=disp, debug=debug, verbose=verbose,
                Rs=Rs, X_init=X, num_stable=num_stable)

            if succeed_L:
                status = "Feasible"
                self.print_status(disp, debug, status, X_L, -np.inf, alpha)
                U = alpha
                step = 2 * step
                num_stable = 5
                X = X_L
            else:
                status = "Infeasible"
                self.print_status(disp, debug, status, X_L, -np.inf, alpha)
                if step > search_tol:
                    step = search_tol
                    num_stable = 15
                else:
                    step = .5 * step
            self.interactive_wait(interactive)

        if step < search_tol:
            succeed = True
        return (U, X, succeed)
