import numpy as np
from mixtape._reversibility import reversible_transmat
from mixtape.mslds_solvers.sparse_sdp.utils import get_entries, set_entries
from mixtape.mslds_solvers.A_problems import A_problem
from mixtape.mslds_solvers.Q_problems import Q_problem
from mixtape.mslds_solvers.A_problems_no_stability import A_problem_no_stability 
from mixtape.mslds_solvers.Q_problems_no_stability import Q_problem_no_stability
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
        self.a_prob = A_problem(n_features)
        self.q_prob = Q_problem(n_features)
        self.a_prob_no_stability = A_problem_no_stability(n_features)
        self.q_prob_no_stability = Q_problem_no_stability(n_features)

    def do_hmm_mstep(self, stats):
        print "Starting hmm mstep"
        print "Starting transmat update"
        transmat = self.transmat_solve(stats)
        print "Starting means update"
        means = self.means_update(stats)
        print "Starting covars update"
        covars = self.covars_update(means, stats)
        print "Done with hmm-mstep"
        return transmat, means, covars

    def do_mstep(self, As, Qs, bs, means, covars, stats, N_iter=500,
                    N_iter_short=20, N_iter_long=40,
                    verbose=False, gamma=.5, tol=1e-1, num_biconvex=1,
                    search_tol=1e-1, stable=True):
        # Remove these copies once the memory error is isolated.
        covars = np.copy(covars)
        means = np.copy(means)
        As = np.copy(As)
        Qs = np.copy(Qs)
        bs = np.copy(bs)
        transmat = self.transmat_solve(stats)
        A_upds, Q_upds, b_upds = self.AQb_update(As, Qs, bs,
                means, covars, stats, N_iter=N_iter,
                N_iter_short=N_iter_short, N_iter_long=N_iter_long,
                verbose=verbose, gamma=gamma, tol=tol,
                num_biconvex=num_biconvex, search_tol=search_tol,
                stable=stable)
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
                min_eig = np.amin(np.linalg.eigh(covar)[0])
                if min_eig < 0:
                    # Assume min_eig << 1
                    covar_new = covar + (2*abs(min_eig) *
                                            np.eye(self.n_features))
                    covar = covar_new
                covar += (1e-5) * np.eye(self.n_features)
                covars.append(covar)
            else:
                # Almost no evidence. Set to identity in absence of other
                # info to prevent Cholesky factorization from crasing.
                covars.append(np.eye(self.n_features))
        return covars

    def print_aux_matrices(self, Bs, Cs, Es, Ds, Fs, gs):
        # TODO: make choice of aux output automatic
        np.set_printoptions(threshold=np.nan)
        with open("aux_matrices.txt", 'w') as f:
            display_string = """
            ++++++++++++++++++++++++++
            Current Aux Matrices.
            ++++++++++++++++++++++++++
            """
            for i in range(self.n_components):
                display_string += ("""
                --------
                State %d
                --------
                """ % i)
                display_string += (("\nBs[%d]:\n"%i + str(Bs[i]) + "\n")
                                 + ("\nCs[%d]:\n"%i + str(Cs[i]) + "\n")
                                 + ("\nDs[%d]:\n"%i + str(Ds[i]) + "\n")
                                 + ("\nEs[%d]:\n"%i + str(Es[i]) + "\n")
                                 + ("\nFs[%d]:\n"%i + str(Fs[i]) + "\n")
                                 + ("\ngs[%d]:\n"%i + str(gs[i]) + "\n"))
            display_string = (bcolors.WARNING + display_string
                                + bcolors.ENDC)
            f.write(display_string)
        np.set_printoptions(threshold=1000)


    def means_update(self, stats):
        means = (stats['obs']) / (stats['post'][:, np.newaxis])
        return means

    def AQb_update(self, As, Qs, bs, means, covars, stats, N_iter=500,
                    N_iter_short=20, N_iter_long=40,
                    verbose=False, gamma=.5, tol=1e-1, num_biconvex=2,
                    search_tol=1e-1, stable=True):
        Bs, Cs, Es, Ds, Fs, gs = self.compute_aux_matrices(As, bs, 
                                                        covars, stats)
        self.print_aux_matrices(Bs, Cs, Es, Ds, Fs, gs)
        A_upds, Q_upds, b_upds = [], [], []

        for i in range(self.n_components):
            B, C, D, E, F, g = Bs[i], Cs[i], Ds[i], Es[i], Fs[i], gs[i]
            A, Q, mu = As[i], Qs[i], means[i]
            if stable:
                A_upd, Q_upd, b_upd = self.AQb_solve(A, Q, mu, B,
                        C, D, E, F, g, i, N_iter=N_iter, 
                        N_iter_short=N_iter_short,
                        N_iter_long=N_iter_long, verbose=verbose,
                        gamma=gamma, tol=tol, num_biconvex=num_biconvex,
                        search_tol=search_tol)
            else:
                A_upd, Q_upd, b_upd = self.AQb_solve_no_stability(A, Q, mu, B,
                        C, D, E, F, g, i, N_iter=N_iter, 
                        N_iter_short=N_iter_short,
                        N_iter_long=N_iter_long, verbose=verbose,
                        gamma=gamma, tol=tol, num_biconvex=num_biconvex,
                        search_tol=search_tol)
            A_upds += [A_upd]
            Q_upds += [Q_upd]
            b_upds += [b_upd]
        return A_upds, Q_upds, b_upds

    def print_banner(self, Type):
        display_string = """
        ##################
        %s SOLVE STARTED
        ##################
        """%Type
        display_string = bcolors.HEADER + display_string + bcolors.ENDC
        print display_string

    def AQb_solve(self, A, Q, mu, B, C, D, E, F, g, iteration,
        interactive=False, disp=True, verbose=False, debug=False, N_iter=500,
        N_iter_short=20, N_iter_long=40, gamma=.5, tol=1e-1, num_biconvex=2,
        search_tol=1e-1):
        for i in range(num_biconvex):
            self.print_banner("Q-%d"%iteration)
            Q_upd = self.q_prob.solve(A, D, F, g, interactive=interactive,
                        disp=disp, debug=debug, verbose=verbose,
                        gamma=gamma, tol=tol, N_iter=N_iter,
                        N_iter_short=N_iter_short, N_iter_long=N_iter_long,
                        search_tol=search_tol)
            if Q_upd != None:
                Q = Q_upd
            else:
                self.q_prob.print_test_case("autogen_Q_tests.py", 
                    A, D, F)
            self.print_banner("A-%d"%iteration)
            A_upd = self.a_prob.solve(B, C, D, E, Q, A_init=A,
                interactive=interactive, disp=disp, debug=debug, N_iter=N_iter,
                N_iter_short=N_iter_short, N_iter_long=N_iter_long,
                verbose=verbose, tol=tol, search_tol=search_tol)
            if A_upd != None:
                A = A_upd
            else:
                self.a_prob.print_test_case("autogen_A_tests.py", 
                    B, C, D, E, Q)
        b = self.b_solve(A, mu)
        return A, Q, b

    def AQb_solve_no_stability(self, A, Q, mu, B, C, D, E, F, g, iteration,
        interactive=False, disp=True, verbose=False, debug=False, N_iter=500,
        N_iter_short=20, N_iter_long=40, gamma=.5, tol=1e-1, num_biconvex=2,
        search_tol=1e-1):
        for i in range(num_biconvex):
            self.print_banner("Q-%d"%iteration)
            Q_upd = self.q_prob_no_stability.solve(A, D, F, g, 
                        interactive=interactive,
                        disp=disp, debug=debug, verbose=verbose,
                        tol=tol, N_iter=N_iter,
                        N_iter_short=N_iter_short, N_iter_long=N_iter_long,
                        search_tol=search_tol)
            if Q_upd != None:
                Q = Q_upd
            else:
                self.q_prob.print_test_case("autogen_Q_tests.py", 
                    A, D, F)
            self.print_banner("A-%d"%iteration)
            d = .9
            A_upd = self.a_prob_no_stability.solve(d, B, C, E, Q, A_init=A,
                interactive=interactive, disp=disp, debug=debug, N_iter=N_iter,
                N_iter_short=N_iter_short, N_iter_long=N_iter_long,
                verbose=verbose, tol=tol, search_tol=search_tol)
            if A_upd != None:
                A = A_upd
            else:
                self.a_prob.print_test_case("autogen_A_tests.py", 
                    B, C, D, E, Q)
        b = self.b_solve(A, mu)
        return A, Q, b

    def b_solve(self, A, mu):
        b =  mu - np.dot(A, mu)
        return b

    # FIX ME!
    def transmat_solve(self, stats):
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
        return revised_counts

    def compute_aux_matrices(self, As, bs, covars, stats):
        n_components = self.n_components
        n_features = self.n_features
        Bs, Cs, Es, Ds, Fs, gs = [], [], [], [], [], []
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
            g = stats['post'][i]
            Bs += [B]
            Cs += [C]
            Es += [E]
            Ds += [D]
            Fs += [F]
            gs += [g]
        return Bs, Cs, Es, Ds, Fs, gs

