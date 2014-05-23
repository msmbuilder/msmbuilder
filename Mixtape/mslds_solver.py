import numpy as np
from mixtape._reversibility import reversible_transmat
from mixtape.mslds_solvers.sparse_sdp.constraints import A_constraints
from mixtape.mslds_solvers.sparse_sdp.constraints import A_coords
from mixtape.mslds_solvers.sparse_sdp.constraints import Q_constraints
from mixtape.mslds_solvers.sparse_sdp.constraints import Q_coords
from mixtape.mslds_solvers.sparse_sdp.objectives import A_dynamics
from mixtape.mslds_solvers.sparse_sdp.objectives import grad_A_dynamics
from mixtape.mslds_solvers.sparse_sdp.objectives import log_det_tr
from mixtape.mslds_solvers.sparse_sdp.objectives import grad_log_det_tr
from mixtape.mslds_solvers.sparse_sdp.general_sdp_solver \
        import GeneralSolver
from mixtape.mslds_solvers.sparse_sdp.utils import get_entries, set_entries
from mixtape.utils import bcolors
import pickle

class MetastableSwitchingLDSSolver(object):
    """
    This class should be a functional wrapper that takes in lists of
    parameters As, Qs, bs, covars, means along with sufficient statistics
    and returns updated lists. Not much state should stored.
    """
    def __init__(self, n_components, n_features):
        self.covars_prior = 1e-2
        self.covars_weight = 1.
        self.n_components = n_components
        self.n_features = n_features

    def do_hmm_mstep(self, stats):
        transmat = transmat_solve(stats)
        means = self.means_update(stats)
        covars = self.covars_update(means, stats)
        return transmat, means, covars

    def do_mstep(self, As, Qs, bs, means, covars, stats, N_iter=50):
        # Remove these copies once the memory error is isolated.
        covars = np.copy(covars)
        means = np.copy(means)
        As = np.copy(As)
        Qs = np.copy(Qs)
        bs = np.copy(bs)
        transmat = transmat_solve(stats)
        A_upds, Q_upds, b_upds = self.AQb_update(As, Qs, bs,
                means, covars, stats, N_iter=N_iter)
        return transmat, A_upds, Q_upds, b_upds

    def covars_update(self, means, stats):
        covars = []
        cvweight = max(self.covars_weight - self.n_features, 0)
        for c in range(self.n_components):
            covar = None
            obsmean = np.outer(stats['obs'][c], means[c])

            cvnum = (stats['obs*obs.T'][c]
                        - obsmean - obsmean.T
                        + np.outer(means[c], means[c])
                        * stats['post'][c]) \
                + self.covars_prior * np.eye(self.n_features)
            cvdenom = (cvweight + stats['post'][c])
            if cvdenom > np.finfo(float).eps:
                covar = ((cvnum) / cvdenom)

                # Deal with numerical issues
                # Might be slightly negative due to numerical issues
                min_eig = min(np.linalg.eig(covar)[0])
                if min_eig < 0:
                    # Assume min_eig << 1
                    covar += (2 * abs(min_eig) *
                                        np.eye(self.n_features))
                covars.append(covar)
            else:
                covars.append(np.zeros(np.shape(obsmean)))
        return covars

    def print_aux_matrices(self, Bs, Cs, Es, Ds, Fs):
        # TODO: make choice of aux output automatic
        np.set_printoptions(threshold=np.nan)
        with open("aux_matrices.txt", 'w') as f:
            display_string = """
            ++++++++++++++++++++++++++
            Current Aux Matrices.
            ++++++++++++++++++++++++++
            """
            for i in range(self.n_components):
                B, C, D, E, F = Bs[i], Cs[i], Ds[i], Es[i], Fs[i]
                display_string += ("""
                --------
                State %d
                --------
                """ % i)
                display_string += (("\nBs[%d]:\n"%i + str(Bs[i]) + "\n")
                                 + ("\nCs[%d]:\n"%i + str(Cs[i]) + "\n")
                                 + ("\nDs[%d]:\n"%i + str(Ds[i]) + "\n")
                                 + ("\nEs[%d]:\n"%i + str(Es[i]) + "\n")
                                 + ("\nFs[%d]:\n"%i + str(Fs[i]) + "\n"))
            display_string = (bcolors.WARNING + display_string
                                + bcolors.ENDC)
            f.write(display_string)
        np.set_printoptions(threshold=1000)


    def means_update(self, stats):
        means = (stats['obs']) / (stats['post'][:, np.newaxis])
        return means

    def AQb_update(self, As, Qs, bs, means, covars, stats, N_iter=50):
        Bs, Cs, Es, Ds, Fs = compute_aux_matrices(self.n_components,
                self.n_features, As, bs, covars, stats)
        self.print_aux_matrices(Bs, Cs, Es, Ds, Fs)
        A_upds, Q_upds, b_upds = [], [], []

        for i in range(self.n_components):
            B, C, D, E, F = Bs[i], Cs[i], Ds[i], Es[i], Fs[i]
            A, Q, mu = As[i], Qs[i], means[i]
            A_upd, Q_upd, b_upd = AQb_solve(self.n_features, A, Q, mu, B,
                    C, D, E, F, N_iter=N_iter)
            A_upds += [A_upd]
            Q_upds += [Q_upd]
            b_upds += [b_upd]
        return A_upds, Q_upds, b_upds

def print_Q_test_case(test_file, A, D, F, dim):
    display_string = "Q-solve failed. Autogenerating Q test case"
    display_string = (bcolors.FAIL + display_string
                        + bcolors.ENDC)
    print display_string
    np.set_printoptions(threshold=np.nan)
    with open(test_file, 'a') as f:
        test_string = ""
        test_string += "\ndef Q_solve_test():\n"
        test_string += "\t#Auto-generated test case from failing run of\n"
        test_string += "\t#Q-solve:\n"
        test_string += "\timport numpy as np\n"
        test_string += "\tfrom mixtape.mslds_solver import AQb_solve,"\
                            + " A_solve, Q_solve\n"
        test_string += "\tblock_dim = %d\n"%dim
        test_string += "\tA = (\n\t\tnp." + repr(A) + ")\n"
        test_string += "\tD = (\n\t\tnp." + repr(D) + ")\n"
        test_string += "\tF = (\n\t\tnp." + repr(F) + ")\n"
        test_string += "\tQ_solve(block_dim, A, D, F, \n"
        test_string += "\t\tdisp=True, debug=False, verbose=False,\n"
        test_string += "\t\tRs=[100])\n"
        f.write(test_string)
    np.set_printoptions(threshold=1000)

def print_A_test_case(test_file, B, C, D, E, Q, mu, dim):
    display_string = "A-solve failed. Autogenerating A test case"
    display_string = (bcolors.FAIL + display_string
                        + bcolors.ENDC)
    print display_string
    with open(test_file, 'w') as f:
        test_string = ""
        np.set_printoptions(threshold=np.nan)
        test_string += "\ndef A_solve_test():\n"
        test_string += "\t#Auto-generated test case from failing run of\n"
        test_string += "\t#A-solve:\n"
        test_string += "\timport numpy as np\n"
        test_string += "\timport pickle\n"
        test_string += "\tfrom mixtape.mslds_solver import AQb_solve,"\
                            + " A_solve, Q_solve\n"
        test_string += "\tblock_dim = %d\n"%dim
        pickle.dump(B, open("B.p", "w"))
        test_string += '\tB = pickle.load(open("B.p", "r"))\n'
        pickle.dump(C, open("C.p", "w"))
        test_string += '\tC = pickle.load(open("C.p", "r"))\n'
        pickle.dump(D, open("D.p", "w"))
        test_string += '\tD = pickle.load(open("D.p", "r"))\n'
        pickle.dump(E, open("E.p", "w"))
        test_string += '\tE = pickle.load(open("E.p", "r"))\n'
        pickle.dump(Q, open("Q.p", "w"))
        test_string += '\tQ = pickle.load(open("Q.p", "r"))\n'
        pickle.dump(mu, open("mu.p", "w"))
        test_string += '\tmu = pickle.load(open("mu.p", "r"))\n'
        test_string += "\tA_solve(block_dim, B, C, D, E, Q, mu,\n"
        test_string += "\t\tdisp=True, debug=False, verbose=False,\n"
        test_string += "\t\tRs=[100])\n"
        np.set_printoptions(threshold=1000)
        f.write(test_string)


def AQb_solve(dim, A, Q, mu, B, C, D, E, F, interactive=False, disp=True,
        verbose=False, debug=False, Rs=[10, 100, 1000], N_iter=50):
    # Should this be iterated for biconvex solution? Yes. Need to fix.
    Q_upd = Q_solve(dim, A, D, F, interactive=interactive,
                disp=disp, debug=debug, Rs=Rs)
    if Q_upd != None:
        Q = Q_upd
    else:
        print_Q_test_case("autogen_Q_tests.py", A, D, F, dim)
    A_upd = A_solve(dim, B, C, D, E, Q, mu, interactive=interactive,
                    disp=disp, debug=debug, Rs=Rs, N_iter=N_iter)
    if A_upd != None:
        A = A_upd
    else:
        print_A_test_case("autogen_A_tests.py", B, C, D, E, Q, mu, dim)
    b = b_solve(dim, A, mu)
    return A, Q, b

# FIX ME!
def transmat_solve(stats):
    counts = (np.maximum(stats['trans'], 1e-20).astype(np.float64))
    # Need to fix this......
    #self.transmat_, self.populations_ = \
    #        reversible_transmat(counts)
    (dim, _) = np.shape(counts)
    norms = np.zeros(dim)
    for i in range(dim):
        norms[i] = sum(counts[i])
    revised_counts = np.copy(counts)
    for i in range(dim):
        revised_counts[i] /= norms[i]
        print sum(revised_counts[i])
    #print "counts\n", counts
    #print "revised_counts\n", revised_counts
    return revised_counts

# TEST ME!
def compute_aux_matrices(n_components, n_features, As, bs, covars, stats):
    Bs, Cs, Es, Ds, Fs = [], [], [], [], []
    for i in range(n_components):
        A, b, covar = As[i], bs[i], covars[i]
        b = np.reshape(b, (n_features, 1))
        B = stats['obs*obs[t-1].T'][i]
        mean_but_last = np.reshape(stats['obs[:-1]'][i], (n_features, 1))
        C = np.dot(b, mean_but_last.T)
        E = stats['obs[:-1]*obs[:-1].T'][i]
        D = covars[i]
        F = ((stats['obs[1:]*obs[1:].T'][i]
               - np.dot(stats['obs*obs[t-1].T'][i], A.T)
               - np.dot(np.reshape(stats['obs[1:]'][i],
                                  (n_features, 1)), b.T))
           + (-np.dot(A, stats['obs*obs[t-1].T'][i].T)
               + np.dot(A, np.dot(stats['obs[:-1]*obs[:-1].T'][i], A.T))
               + np.dot(A, np.dot(np.reshape(stats['obs[:-1]'][i],
                                      (n_features, 1)), b.T)))
           + (-np.dot(b, np.reshape(stats['obs[1:]'][i],
                                    (n_features, 1)).T)
               + np.dot(b, np.dot(np.reshape(stats['obs[:-1]'][i],
                                          (n_features, 1)).T, A.T))
               + stats['post[1:]'][i] * np.dot(b, b.T)))
        Bs += [B]
        Cs += [C]
        Es += [E]
        Ds += [D]
        Fs += [F]
    return Bs, Cs, Es, Ds, Fs

def b_solve(n_features, A, mu):
    #print "mu:\n", mu
    b = np.dot(np.eye(n_features) - A, mu)
    #print "b:\n", b
    #return b
    # b = mu since constraint A mu == 0
    return mu

def A_solve(block_dim, B, C, D, E, Q, mu, interactive=False,
        disp=True, verbose=False, debug=False, Rs=[10, 100, 1000],
        N_iter=50):
    """
    Solves A optimization.

    min_A Tr [ Q^{-1} ([C - B] A.T + A [C - B].T + A E A.T]

          --------------------
         | D-Q    A           |
    X =  | A.T  D^{-1}        |
         |              I   A |
         |             A.T  I |
          --------------------
    A mu == 0
    X is PSD
    """
    # Figure out a more elegant way to stop this...
    B = np.copy(B)
    C = np.copy(C)
    D = np.copy(D)
    E = np.copy(E)
    Q = np.copy(Q)
    mu = np.copy(mu)
    # Refactor this better somehow?
    dim = 4*block_dim
    scale_factor = (max(np.linalg.norm(C-B, 2), np.linalg.norm(E,2)))
    C = C/scale_factor
    B = B/scale_factor
    E = E/scale_factor
    print "scale_factor: ", scale_factor
    eps = 1e-4
    tol = 1e-1

    scale = 1./np.sqrt(np.linalg.norm(D, 2))
    print "scale: ", scale
    # Rescaling
    D *= scale
    Q *= scale
    # For numerical stability
    c = 1e-1
    # Compute post-scaled inverses
    Dinv = np.linalg.inv(D+c*np.eye(block_dim))
    Qinv = np.linalg.inv(Q+c*np.eye(block_dim))
    R = np.abs(np.trace(D)) + np.abs(np.trace(Dinv)) + 2 * block_dim
    Rs = [R]
    print "R: ", R
    As, bs, Cs, ds, Fs, gradFs, Gs, gradGs = \
            A_constraints(block_dim, D, Dinv, Q, mu)
    (D_Q_cds, Dinv_cds, I_1_cds, I_2_cds,
        A_1_cds, A_T_1_cds, A_2_cds, A_T_2_cds) = A_coords(block_dim)
    def obj(X):
        return A_dynamics(X, block_dim, C, B, E, Qinv)
    def grad_obj(X):
        return grad_A_dynamics(X, block_dim, C, B, E, Qinv)
    g = GeneralSolver(dim, eps)
    g.save_constraints(obj, grad_obj, As, bs, Cs, ds,
            Fs, gradFs, Gs, gradGs)
    (L, U, X, succeed) = g.solve(N_iter, tol,
            interactive=interactive, disp=disp, verbose=verbose,
            debug=debug, Rs=Rs)
    if succeed:
        A_1 = get_entries(X, A_1_cds)
        A_T_1 = get_entries(X, A_T_1_cds)
        A_2 = get_entries(X, A_2_cds)
        A_T_2 = get_entries(X, A_T_2_cds)
        A = (A_1 + A_T_1 + A_2 + A_T_2) / 4.
        if disp:
            print "A:\n", A
        return A

def Q_solve(block_dim, A, D, F, interactive=False, disp=True,
        verbose=False, debug=False, Rs=[10, 100, 1000]):
    """
    Solves Q optimization.

    min_Q -log det R + Tr(RF)
          --------------
         |D-ADA.T  I    |
    X =  |   I     R    |
         |            R |
          --------------
    X is PSD
    """
    # Figure out more elegant way to stop this.
    D = np.copy(D)
    F = np.copy(F)
    # Refactor this better somehow?
    dim = 3*block_dim
    eps = 1e-4
    tol = 1e-1
    N_iter = 100
    scale = 1./np.amax(np.linalg.eigh(D)[0])
    R = (scale*np.trace(D)
            + 2*(1./scale)*np.trace(np.linalg.inv(D)))
    Rs = [R]
    # Rescaling
    D *= scale
    As, bs, Cs, ds, Fs, gradFs, Gs, gradGs = \
            Q_constraints(block_dim, A, F, D)
    (D_ADA_T_cds, I_1_cds, I_2_cds, R_1_cds, R_2_cds) \
            = Q_coords(block_dim)
    g = GeneralSolver(dim, eps)
    def obj(X):
        return log_det_tr(scale*X, F)
    def grad_obj(X):
        return grad_log_det_tr(scale*X, F)
    g.save_constraints(obj, grad_obj, As, bs, Cs, ds,
            Fs, gradFs, Gs, gradGs)
    (L, U, X, succeed) = g.solve(N_iter, tol, interactive=interactive,
            disp=disp, verbose=verbose, debug=debug, Rs=Rs)
    if succeed:
        R_1 = scale*get_entries(X, R_1_cds)
        R_2 = scale*get_entries(X, R_2_cds)
        R_avg = (R_1 + R_2) / 2.
        Q = np.linalg.inv(R_avg)
        if disp:
            print "Q:\n", Q
        return Q
