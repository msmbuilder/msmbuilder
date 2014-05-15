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

class MetastableSwitchingLDSSolver(object):
    def __init__(self, n_components, n_features):
        self.covars_prior = 1e-2
        self.covars_weight = 1.
        self.n_components = n_components
        self.n_features = n_features

    # Delete this and replace with sklearn?
    #def do_hmm_mstep(self, stats):
    #    self.means_update(stats)
    #    self.covars_update(stats)

    def do_mstep(self, As, Qs, bs, means, covars, stats):
        transmat = transmat_solve(stats)
        A_upds, Q_upds, b_upds = AQb_solve(self.n_components,
                self.n_features, As, Qs, bs, means, covars, stats)
        return transmat, A_upds, Q_upds, b_upds

    def covars_update(self, stats):
        cvweight = max(self.covars_weight - self.n_features, 0)
        for c in range(self.n_components):
            obsmean = np.outer(stats['obs'][c], self.means_[c])

            cvnum = (stats['obs*obs.T'][c]
                        - obsmean - obsmean.T
                        + np.outer(self.means_[c], self.means_[c])
                        * stats['post'][c]) \
                + self.covars_prior * np.eye(self.n_features)
            cvdenom = (cvweight + stats['post'][c])
            if cvdenom > np.finfo(float).eps:
                self.covars_[c] = ((cvnum) / cvdenom)

            # Deal with numerical issues
            # Might be slightly negative due to numerical issues
            min_eig = min(np.linalg.eig(self.covars_[c])[0])
            if min_eig < 0:
                # Assume min_eig << 1
                self.covars_[c] += (2 * abs(min_eig) *
                                    np.eye(self.n_features))

    def means_update(self, stats):
        self.means_ = (stats['obs']) / (stats['post'][:, np.newaxis])

def transmat_solve(stats):
    counts = (np.maximum(stats['trans'], 1e-20).astype(np.float64))
    # Need to fix this......
    #self.transmat_, self.populations_ = \
    #        reversible_transmat(counts)
    (dim, _) = np.shape(counts)
    norms = np.zeros(dim)
    for i in range(dim):
        norms[i] = sum(counts[:,i])
    revised_counts = np.copy(counts)
    for i in range(dim):
        revised_counts[:,i] /= norms[i]
        print sum(revised_counts[:,i])
    print "counts\n", counts
    #print "revised_counts\n", revised_counts
    return revised_counts

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

def AQb_solve(n_components, n_features, As, Qs, bs, means, covars, stats):
    Bs, Cs, Es, Ds, Fs = compute_aux_matrices(n_components, n_features,
            As, bs, covars, stats)
    A_upds, Q_upds, b_upds = [], [], []

    for i in range(n_components):
        # Why don't we use all these terms?
        B, C, E, D, F = Bs[i], Cs[i], Es[i], Ds[i], Fs[i]
        Dinv = np.linalg.inv(D) # Should cache this to save time.
        A, Q, mu = As[i], Qs[i], means[i]
        # Should this be iterated for biconvex solution?
        A_upd = A_solve(n_features, D, Dinv, Q)
        Q_upd = Q_solve(n_features, A, F, D)
        b_upd = b_solve(n_features, A, mu)
        A_upds += [A_upd]
        Q_upds += [Q_upd]
        b_upds += [b_upd]
    return A_upds, Q_upds, b_upds


def b_solve(n_features, A, mu):
     b = np.dot(np.eye(n_features) - A, mu)
     return b

def A_solve(block_dim, D, Dinv, Q):
    # Refactor this better somehow?
    dim = 4*block_dim
    R = 10 # figure out more general R
    L, U = (-10, 10) # figure out more general choice
    eps = 1e-4
    tol = 1e-3
    N_iter = 50
    As, bs, Cs, ds, Fs, gradFs, Gs, gradGs = \
            A_constraints(block_dim, D, Dinv, Q)
    (D_Q_cds, Dinv_cds, I_1_cds, I_2_cds,
        A_1_cds, A_T_1_cds, A_2_cds, A_T_2_cds) = A_coords(dim)
    def obj(X):
        return A_dynamics(X, block_dim, C, B, E, Qinv)
    def grad_obj(X):
        return grad_A_dynamics(X, block_dim, C, B, E, Qinv)
    g = GeneralSolver(R, L, U, dim, eps)
    g.save_constraints(obj, grad_obj, As, bs, Cs, ds,
            Fs, gradFs, Gs, gradGs)
    (alpha, U, X_U, L, X_L, succeed) = g.solve(N_iter, tol,
            interactive=True)
    if succeed:
        A_1 = get_entries(X_L, A_1_cds)
        return A_1

def Q_solve(block_dim, A, F, D):
    # Refactor this better somehow?
    dim = 3*block_dim
    R = 10 # figure out more general R
    L, U = (-10, 10) # figure out more general choice
    eps = 1e-4
    tol = 1e-3
    N_iter = 50
    As, bs, Cs, ds, Fs, gradFs, Gs, gradGs = \
            Q_constraints(block_dim, A, F, D)
    (D_ADA_T_cds, I_1_cds, I_2_cds, R_1_cds, R_2_cds) \
            = Q_coords(block_dim)
    g = GeneralSolver(R, L, U, dim, eps)
    def obj(X):
        return log_det_tr(X, F)
    def grad_obj(X):
        return grad_log_det_tr(X, F)
    g.save_constraints(obj, grad_obj, As, bs, Cs, ds,
            Fs, gradFs, Gs, gradGs)
    (alpha, U, X_U, L, X_L, succeed) = g.solve(N_iter, tol,
            interactive=True)
    if succeed:
        R_1 = get_entries(X_L, R_1_cds)
        Q_1 = np.linalg.inv(R_1)
        return Q_1
