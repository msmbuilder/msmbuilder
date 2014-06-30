import numpy as np
from mixtape.utils import print_solve_test_case
from mixtape.mslds_solvers.sparse_sdp.utils import get_entries, set_entries
from mixtape.mslds_solvers.sparse_sdp.constraints import many_batch_equals
from mixtape.mslds_solvers.sparse_sdp.constraints import grad_many_batch_equals
from mixtape.mslds_solvers.sparse_sdp.general_sdp_solver \
    import GeneralSolver
from mixtape.utils import bcolors

class A_problem_no_stability(object):

    def __init__(self, dim):
        self.dim = dim
        self.scale = 2

    # Tr [ Q^{-1} ([C - B] A.T + A [C - B].T + A E A.T]
    def objective(self, X, C, B, E, Qinv):
        """
        min Tr [ Q^{-1} ([C - B] A.T + A [C - B].T + A E A.T]

              --------
             |  dI  A |
        X =  | A.T  I |
              --------
        X is PSD
        """
        (d_I_cds, I_cds, A_cds, A_T_cds) = self.coords()

        A = get_entries(X, A_cds)
        A_T = get_entries(X, A_T_cds)
        term1 = np.dot(Qinv, (np.dot(C-B, A.T) + np.dot(C-B, A.T).T
                            + np.dot(A, np.dot(E, A.T))))
        term2 = np.dot(Qinv, (np.dot(C-B, A_T) + np.dot(C-B, A_T).T
                            + np.dot(A_T.T, np.dot(E, A_T))))
        return (np.trace(term1) + np.trace(term2))

    # grad Tr [Q^{-1} (C - B) A.T] = Q^{-1} (C - B)
    # grad Tr [Q^{-1} A [C - B].T] = Q^{-T} (C - B)
    # grad Tr [Q^{-1} A E A.T] = Q^{-T} A E.T + Q^{-1} A E
    def grad_objective(self, X, C, B, E, Qinv):
        """
        min Tr [ Q^{-1} ([C - B] A.T + A [C - B].T + A E A.T]

              --------
             | dI   A |
        X =  | A.T  I |
              --------
        X is PSD
        """
        (d_I_cds, I_cds, A_cds, A_T_cds) = self.coords()
        grad = np.zeros(np.shape(X))
        A = get_entries(X, A_cds)
        A_T = get_entries(X, A_T_cds)

        gradA = (np.dot(Qinv, C-B) + np.dot(Qinv.T, C-B) 
                    + np.dot(Qinv.T, np.dot(A, E.T)) 
                    + np.dot(Qinv, np.dot(A, E)) )
        gradA_T = (np.dot(Qinv, C-B) + np.dot(Qinv.T, C-B) 
                    + np.dot(Qinv.T, np.dot(A_T.T, E.T)) 
                    + np.dot(Qinv, np.dot(A_T.T, E))).T

        set_entries(grad, A_cds, gradA)
        set_entries(grad, A_T_cds, gradA_T)
        return grad

    def coords(self):
        dim = self.dim
        """
         --------
        | dI   A |
        | A.T  I |
         --------
        """
        d_I_cds = (0, dim, 0, dim)
        I_cds = (dim, 2*dim, dim, 2*dim)


        """
              --------
             | dI   A |
        X =  | A.T  I |
              --------
          ----------
         |  _     A |
         | A.T    _ |
          ----------
        """
        A_cds = (0, dim, dim, 2*dim)
        A_T_cds = (dim, 2*dim, 0, dim)

        return (d_I_cds, I_cds, A_cds, A_T_cds)

    def constraints(self, d):

        As, bs, Cs, ds, = [], [], [], []
        Fs, gradFs, Gs, gradGs = [], [], [], []

        (d_I_cds, I_cds, A_cds, A_T_cds) = self.coords()

        constraints = []

        """
        We need to enforce constant equalities in X
              --------
             | dI   _ |
        X =  |  _   I |
              --------
        """
        constraints += [(d_I_cds, d*np.eye(self.dim)), (I_cds,
        np.eye(self.dim))]

        # Add constraints to Gs
        def const_regions(X):
            return many_batch_equals(X, constraints)
        def grad_const_regions(X):
            return grad_many_batch_equals(X, constraints)
        Gs.append(const_regions)
        gradGs.append(grad_const_regions)

        return As, bs, Cs, ds, Fs, gradFs, Gs, gradGs

    def print_fail_banner(self):
        display_string = """
        ###############
        NOT ENOUGH DATA
        ###############
        """
        display_string = bcolors.FAIL + display_string + bcolors.ENDC
        print display_string

    def solve(self, d, B, C, E, Q, A_init=None, interactive=False, disp=True,
        verbose=False, debug=False, N_iter=500, N_iter_short=20,
        N_iter_long=40, search_tol=1e-1, tol=1e-1, min_step_size=1e-6,
        methods=['frank_wolfe']):
        """
        Solves A optimization.

        min_A Tr [ Q^{-1} ([C - B] A.T + A [C - B].T + A E A.T]
              --------
             | dI   A |
        X =  | A.T  I |
              --------
        X is PSD
        """
        prob_dim = 2 * self.dim

        # Copy in inputs 
        B = np.copy(B)
        C = np.copy(C)
        E = np.copy(E)

        # Scale down objective matrices 
        scale_factor = (max(np.linalg.norm(C-B, 2), np.linalg.norm(E,2)))
        if scale_factor < 1e-6:
            # If A has no observations, not much we can say
            self.print_fail_banner()
            return None 

        C = C/scale_factor
        B = B/scale_factor
        E = E/scale_factor

        # Numerical stability 
        scale = 1./np.linalg.norm(Q, 2)

        # Rescaling
        Q *= scale

        # Improving conditioning
        delta = 1e-2
        Q = Q + delta*np.eye(self.dim)
        Qinv = np.linalg.inv(Q)

        # Compute trace upper bound
        R = (d+1)*self.dim 
        Rs = [R]

        As, bs, Cs, ds, Fs, gradFs, Gs, gradGs = \
                self.constraints(d)
        (d_I_cds, I_cds, A_cds, A_T_cds) = self.coords()

        # Construct init matrix
        if True:
            X_init = np.zeros((prob_dim, prob_dim))
            set_entries(X_init, d_I_cds, d*np.eye(self.dim))
            #set_entries(X_init, A_cds, A_init)
            #set_entries(X_init, A_T_cds, A_init.T)
            set_entries(X_init, I_cds, np.eye(self.dim))
            min_eig = np.amin(np.linalg.eigh(X_init)[0])
            if min_eig < 0:
                # X_init may have small negative eigenvalues
                X_init += 2*np.abs(min_eig)*np.eye(prob_dim)
        else:
            X_init = None

        def obj(X):
            return self.objective(X, C, B, E, Qinv)
        def grad_obj(X):
            return self.grad_objective(X, C, B, E, Qinv)

        g = GeneralSolver()
        g.save_constraints(prob_dim, obj, grad_obj, As, bs, Cs, ds,
                Fs, gradFs, Gs, gradGs)
        (U, X, succeed) = g.solve(N_iter, tol, search_tol,
                N_iter_short=N_iter_short, N_iter_long=N_iter_long,
                interactive=interactive, disp=disp, verbose=verbose,
                debug=debug, Rs=Rs, min_step_size=min_step_size,
                methods=methods, X_init=X_init)
        if succeed:
            A = get_entries(X, A_cds)
            norm = np.linalg.norm(A, 2)
            # In high dimensions, sometimes numerical issues intrude
            if norm >= 1:
                A *= (d/norm)
            if A_init != None:
                X_init = np.zeros((prob_dim, prob_dim))
                set_entries(X_init, A_cds, A_init)
                set_entries(X_init, A_T_cds, A_init.T)
                print "obj(X_init) = %f"%obj(X_init)
                print "obj(X) = %f"%obj(X)
                if obj(X_init) <= U:
                    return A_init
                else:
                    return A
            else:
                return A

    def print_test_case(self, test_file, B, C, D, E, Q):
        matrices = [(B, "B"), (C, "C"), (D, "D"), (E, "E"), (Q, "Q")]
        print_solve_test_case("A", matrices, self.dim, test_file)
