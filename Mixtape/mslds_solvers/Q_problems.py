import numpy as np
from numpy.linalg import LinAlgError
from mixtape.utils import print_solve_test_case
from mixtape.mslds_solvers.sparse_sdp.utils import get_entries, set_entries
from mixtape.mslds_solvers.sparse_sdp.constraints import many_batch_equals
from mixtape.mslds_solvers.sparse_sdp.constraints import grad_many_batch_equals
from mixtape.mslds_solvers.sparse_sdp.general_sdp_solver \
    import GeneralSolver
from mixtape.utils import bcolors

class Q_problem(object):

    def __init__(self, dim):
        self.dim = dim
        self.scale = 2
        pass

    # - G log det R + Tr(RF)
    def objective(self, X, F, G):
        """
        minimize -G log det R + Tr(RF)
              -----------
             |D-ADA.T  cI |
        X =  |   cI     R |
              -----------
        X is PSD
        """
        np.seterr(divide='raise')
        (D_ADA_T_cds, I_1_cds, I_2_cds, R_cds) = self.coords()
        R = get_entries(X, R_cds)
        try:
            L = np.linalg.cholesky(R)
            log_det = 2*np.sum(np.log(np.diag(L)))
            val = -G * log_det + np.trace(np.dot(R, F))
        except (FloatingPointError, LinAlgError) as e:
            val = np.inf
        return val 

    # grad - Glog det R = - GR^{-1}.T = - GQ.T (see Boyd and Vandenberge, A4.1)
    # grad tr(RF) = F^T
    def grad_objective(self, X, F, G):
        """
        minimize -G log det R + Tr(RF)
              -----------
             |D-ADA.T  cI |
        X =  |   cI     R |
              -----------
        X is PSD
        """
        (dim, _) = np.shape(X)
        (D_ADA_T_cds, I_1_cds, I_2_cds, R_cds) = self.coords()
        grad = np.zeros(np.shape(X))
        R = get_entries(X, R_cds)
        # Need to avoid ill-conditioning of R1, R2
        R = R + (1e-5) * np.eye(self.dim)
        Q = np.linalg.inv(R)
        gradR = -G * Q.T + F.T
        set_entries(grad, R_cds, gradR)
        return grad


    def coords(self):
        """
        Helper function that specifies useful coordinates for
        the Q convex program.
        minimize -log det R + Tr(RB)
              -----------
             |D-ADA.T  cI |
        X =  |   cI     R |
              -----------
        X is PSD
        """
        dim = self.dim
        # Block 1
        D_ADA_T_cds = (0, dim, 0, dim)
        I_1_cds = (0, dim, dim, 2*dim)
        I_2_cds = (dim, 2*dim, 0, dim)
        R_cds = (dim, 2*dim, dim, 2*dim)

        return (D_ADA_T_cds, I_1_cds, I_2_cds, R_cds)

    def constraints(self, A, B, D, c):
        """
        Specifies the convex program required for Q optimization.

        minimize -log det R + Tr(RB)
              -----------
             |D-ADA.T  cI |
        X =  |   cI     R |
              -----------
        X is PSD
        """
        dim = self.dim
        constraints = []

        (D_ADA_T_cds, I_1_cds, I_2_cds, R_cds) = self.coords()
        As, bs, Cs, ds, = [], [], [], []
        Fs, gradFs, Gs, gradGs = [], [], [], []

        """
        We need to enforce constant equalities in X.
          ------------
         |D-ADA.T   cI |
    C =  | cI        _ |
          ------------
        """
        D_ADA_T = D - np.dot(A, np.dot(D, A.T))
        constraints += [(D_ADA_T_cds, D_ADA_T), (I_1_cds, c*np.eye(dim)),
                        (I_2_cds, c*np.eye(dim))]

        # Add constraints to Gs
        def const_regions(X):
            return many_batch_equals(X, constraints)
        def grad_const_regions(X):
            return grad_many_batch_equals(X, constraints)
        Gs.append(const_regions)
        gradGs.append(grad_const_regions)

        """ We need to enforce linear inequalities
              -----
             |-  - |
        C =  |-  R |
              -----
        """
        return As, bs, Cs, ds, Fs, gradFs, Gs, gradGs

    def print_fail_banner(self):
        display_string = """
        ###############
        NOT ENOUGH DATA
        ###############
        """
        display_string = bcolors.FAIL + display_string + bcolors.ENDC
        print display_string

    def solve(self, A, D, F, G, interactive=False, disp=True,
            verbose=False, debug=False, Rs=[10, 100, 1000], N_iter=400,
            N_iter_short=20, N_iter_long=40,
            gamma=.5, tol=1e-1, search_tol=1e-1, min_step_size=1e-6,
            methods=['frank_wolfe']):
        """
        Solves Q optimization.

        min_Q -G log det R + Tr(RF)
              -----------
             |D-ADA.T  cI |
        X =  |   cI     R |
              -----------
        X is PSD
        """
        dim = self.dim
        prob_dim = self.scale*self.dim

        # Copy over initial data 
        D = np.copy(D)
        F = np.copy(F)
        c = np.sqrt(1./gamma)

        # Numerical stability 
        scale = 1./np.linalg.norm(D,2)

        # Rescaling
        D *= scale

        # Scale down objective matrices
        scale_factor = max(np.linalg.norm(F, 2), np.abs(G))
        if scale_factor < 1e-6:
            # F can be zero if there are no observations for this state 
            self.print_fail_banner()
            return None 

        # Improving conditioning
        delta=1e-4
        D = D + delta*np.eye(dim)
        Dinv = np.linalg.inv(D)
        D_ADA_T = D - np.dot(A, np.dot(D, A.T)) 

        # Sometimes D_ADA_T has small negative eigenvalues
        min_eig = np.amin(np.linalg.eigh(D_ADA_T)[0])
        if min_eig < 0:
            # assume min_eig is small
            D_ADA_T += 2*np.abs(min_eig)*np.eye(dim)
        

        # Compute trace upper bound
        R = (2*np.trace(D) + 2*(1./gamma)*np.trace(Dinv))
        Rs = [R]

        As, bs, Cs, ds, Fs, gradFs, Gs, gradGs = \
                self.constraints(A, F, D, c)
        (D_ADA_T_cds, I_1_cds, I_2_cds, R_cds) = self.coords()

        # Construct init matrix
        X_init = np.zeros((prob_dim, prob_dim))
        set_entries(X_init, D_ADA_T_cds, D_ADA_T)
        set_entries(X_init, I_1_cds, c*np.eye(dim))
        set_entries(X_init, I_2_cds, c*np.eye(dim))
        #Qinv_init = np.linalg.inv(D_ADA_T) 
        Qinv_init = (c**2) * np.linalg.inv(D_ADA_T) 
        set_entries(X_init, R_cds, Qinv_init)
        min_eig = np.amin(np.linalg.eigh(X_init)[0])
        if min_eig < 0:
            # X_init may have small negative eigenvalues
            X_init += 2*np.abs(min_eig)*np.eye(prob_dim)

        g = GeneralSolver()
        def obj(X):
            return (1./scale_factor) * self.objective(X, F, G)
        def grad_obj(X):
            return (1./scale_factor) * self.grad_objective(X, F, G)
        g.save_constraints(prob_dim, obj, grad_obj, As, bs, Cs, ds,
                Fs, gradFs, Gs, gradGs)
        (U, X, succeed) = g.solve(N_iter, tol, search_tol,
            N_iter_short=N_iter_short, N_iter_long=N_iter_long,
            interactive=interactive, disp=disp, verbose=verbose, 
            debug=debug, Rs=Rs, min_step_size=min_step_size,
            methods=methods, X_init=X_init)
        if succeed:
            R = scale*get_entries(X, R_cds)
            # Ensure stability
            R = R + (1e-3) * np.eye(dim)
            Q = np.linalg.inv(R)
            # sometimes Q may have small negative eigenvalues
            min_eig = np.amin(np.linalg.eigh(Q)[0])
            if min_eig < 0:
                Q += 2*np.abs(min_eig)*np.eye(dim)
            return Q

    def print_test_case(self, test_file, A, D, F):
        matrices = [(A, "A"), (D, "D"), (F, "F")]
        print_solve_test_case("Q", matrices, self.dim, test_file)
