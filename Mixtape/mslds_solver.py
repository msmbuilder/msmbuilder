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

    # TEST ME!
    def do_hmm_mstep(self, stats):
        self.means_update(stats)
        self.covars_update(stats)

    def do_mstep(self, As, Qs, bs, means, covars, stats):
        transmat = transmat_solve(stats)
        A_upds, Q_upds, b_upds = self.AQb_update(self.n_components,
                self.n_features, As, Qs, bs, means, covars, stats)
        return transmat, A_upds, Q_upds, b_upds

    # TEST ME!
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

    # TEST ME!
    def means_update(self, stats):
        self.means_ = (stats['obs']) / (stats['post'][:, np.newaxis])

    def AQb_update(self, n_components, n_features,
            As, Qs, bs, means, covars, stats):
        Bs, Cs, Es, Ds, Fs = compute_aux_matrices(n_components, n_features,
                As, bs, covars, stats)
        A_upds, Q_upds, b_upds = [], [], []

        for i in range(n_components):
            # Why don't we use all these terms?
            B, C, D, E, F = Bs[i], Cs[i], Ds[i], Es[i], Fs[i]
            Dinv = np.linalg.inv(D) # Should cache this to save time.
            A, Q, mu = As[i], Qs[i], means[i]
            Qinv = np.linalg.inv(Q)
            A_upd, Q_upd, b_upd = AQb_solve(self.n_features, A, Q, Qinv,
                    mu, B, C, D, Dinv, E, F)
            A_upds += [A_upd]
            Q_upds += [Q_upd]
            b_upds += [b_upd]
        return A_upds, Q_upds, b_upds

def AQb_solve(dim, A, Q, Qinv, mu, B, C, D, Dinv, E, F):
    # Should this be iterated for biconvex solution?
    Q_upd = Q_solve(dim, A, D, Dinv, F)
    A_upd = A_solve(dim, B, C, D, Dinv, E, Q, Qinv)
    b_upd = b_solve(dim, A, mu)
    print "Q_upd: ", Q_upd
    print "A_upd: ", A_upd
    print "b_upd: ", b_upd

# FIX ME!
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
     b = np.dot(np.eye(n_features) - A, mu)
     return b

def A_solve(block_dim, B, C, D, Dinv, E, Q, Qinv):
    """
    Solves A optimization.

    min_A Tr [ Q^{-1} ([C - B] A.T + A [C - B].T + A E A.T]

          --------------------
         | D-Q    A           |
    X =  | A.T  D^{-1}        |
         |              I   A |
         |             A.T  I |
          --------------------
    X is PSD
    """
    # Refactor this better somehow?
    dim = 4*block_dim
    R = np.trace(D) + np.trace(Dinv) + 2 * block_dim
    scale_factor = (max(np.linalg.norm(C-B, 2), np.linalg.norm(E,2)))
    C = C/scale_factor
    B = B/scale_factor
    E = E/scale_factor
    U = np.linalg.norm(Qinv)
    L, U = (-10, 10)
    eps = 1e-4
    tol = 1e-2
    N_iter = 50
    scale = 10.
    # Rescaling
    D *= scale
    Q *= scale
    Dinv *= (1./scale)
    Qinv *= (1./scale)
    As, bs, Cs, ds, Fs, gradFs, Gs, gradGs = \
            A_constraints(block_dim, D, Dinv, Q)
    (D_Q_cds, Dinv_cds, I_1_cds, I_2_cds,
        A_1_cds, A_T_1_cds, A_2_cds, A_T_2_cds) = A_coords(block_dim)
    def obj(X):
        return A_dynamics(X, block_dim, C, B, E, Qinv)
    def grad_obj(X):
        return grad_A_dynamics(X, block_dim, C, B, E, Qinv)
    g = GeneralSolver(R, L, U, dim, eps)
    g.save_constraints(obj, grad_obj, As, bs, Cs, ds,
            Fs, gradFs, Gs, gradGs)
    (alpha, U, X_U, L, X_L, succeed) = g.solve(N_iter, tol,
            interactive=False, disp=True, verbose=False)
    if succeed:
        import pdb
        pdb.set_trace()
        A_1 = R*get_entries(X_L, A_1_cds)
        A_T_1 = R*get_entries(X_L, A_T_1_cds)
        A_2 = R*get_entries(X_L, A_2_cds)
        A_T_2 = R*get_entries(X_L, A_T_2_cds)
        A = (A_1 + A_T_1 + A_2 + A_T_2) / 4.
        return A

def Q_solve(block_dim, A, D, Dinv, F):
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
    # Refactor this better somehow?
    dim = 3*block_dim
    L, U = (0, 1000)
    eps = 1e-4
    tol = 1e-2
    N_iter = 100
    scale = 10.
    R = 10
    # Rescaling
    D *= scale
    As, bs, Cs, ds, Fs, gradFs, Gs, gradGs = \
            Q_constraints(block_dim, A, F, D)
    (D_ADA_T_cds, I_1_cds, I_2_cds, R_1_cds, R_2_cds) \
            = Q_coords(block_dim)
    g = GeneralSolver(R, L, U, dim, eps)
    def obj(X):
        return log_det_tr(scale*X, F)
    def grad_obj(X):
        return grad_log_det_tr(scale*X, F)
    g.save_constraints(obj, grad_obj, As, bs, Cs, ds,
            Fs, gradFs, Gs, gradGs)
    (alpha, U, X_U, L, X_L, succeed) = g.solve(N_iter, tol,
            interactive=False, disp=True, verbose=True)
    if succeed:
        R_1 = scale*R*get_entries(X_L, R_1_cds)
        R_2 = scale*R*get_entries(X_L, R_2_cds)
        R_avg = (R_1 + R_2) / 2.
        Q = np.linalg.inv(R_avg)
        return Q
