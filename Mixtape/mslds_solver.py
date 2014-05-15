class MetastableSwitchingLDSSolver(object):
    def __init__(self):
        self.covars_prior = 1e-2
        self.covars_weight = 1.
        self.transmat_prior = 1.0

    def _do_mstep(self, stats, params, iteration):
        if iteration < self.n_hotstart:
            if 'm' in params:
                self._means_update(stats)
            if 'c' in params:
                self._covars_update(stats)
        if 't' in params:
            self._transmat_update(stats)
        if 'a' in params:
            self._A_update(stats)
        if 'q' in params:
            self._Q_update(stats)
        if 'b' in params:
            self._b_update(stats)

    def _covars_update(self, stats):
        cvweight = max(self.covars_weight - self.n_features, 0)
        for c in range(self.n_states):
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

    def _means_update(self, stats):
        self.means_ = (stats['obs']) / (stats['post'][:, np.newaxis])

    def _transmat_update(self, stats):
        counts = (np.maximum(stats['trans']
                        + self.transmat_prior - 1.0, 1e-20)
                    .astype(np.float64))
        self.transmat_, self.populations_ = \
                _reversibility.reversible_transmat(counts)

    def compute_auxiliary_matrices(self, stats):
        b = np.reshape(self.bs_[i], (self.n_features, 1))
        B = stats['obs*obs[t-1].T'][i]
        Bp = ((stats['obs[1:]*obs[1:].T'][i]
              - np.dot(stats['obs*obs[t-1].T'][i], A.T)
              - np.dot(np.reshape(stats['obs[1:]'][i],
                                  (self.n_features, 1)), b.T))
             + (-np.dot(A, stats['obs*obs[t-1].T'][i].T) +
                np.dot(A, np.dot(stats['obs[:-1]*obs[:-1].T'][i],
                                 A.T)) +
                np.dot(A, np.dot(np.reshape(stats['obs[:-1]'][i],
                                        (self.n_features, 1)), b.T)))
             + (-np.dot(b, np.reshape(stats['obs[1:]'][i],
                                      (self.n_features, 1)).T) +
                np.dot(b, np.dot(np.reshape(stats['obs[:-1]'][i],
                                            (self.n_features, 1)).T,
                                 A.T)) +
                stats['post[1:]'][i] * np.dot(b, b.T)))
        mean_but_last = np.reshape(stats['obs[:-1]'][i],
                                   (self.n_features, 1))
        C = np.dot(b, mean_but_last.T)
        E = stats['obs[:-1]*obs[:-1].T'][i]
        Sigma = self.covars_[i]

    def _A_update(self, stats):
        for i in range(self.n_states):
            sol, _, G, _ = solve_A(self.n_features, B, C, E, Sigma, Q,
                    self.max_iters, self.display_solver_output)
            avec = np.array(sol['x'])
            avec = avec[int(1 + self.n_features * (self.n_features + 1) /
                2):]
            A = np.reshape(avec, (self.n_features, self.n_features),
                           order='F')
            self.As_[i] = A

    def _Q_update(self, stats):
        for i in range(self.n_states):
            A = self.As_[i]
            b = np.reshape(self.bs_[i], (self.n_features, 1))
            sol, _, _, _ = solve_Q(self.n_features, A, B, Sigma,
                    self.max_iters, self.display_solver_output)
            qvec = np.array(sol['x'])
            qvec = qvec[int(1 + self.n_features
                                    * (self.n_features + 1) / 2):]
            Q = np.zeros((self.n_features, self.n_features))
            for j in range(self.n_features):
                for k in range(j + 1):
                    vec_pos = int(j * (j + 1) / 2 + k)
                    Q[j, k] = qvec[vec_pos]
                    Q[k, j] = Q[j, k]
            self.Qs_[i] = Q

    def _b_update(self, stats):
        for i in range(self.n_states):
            mu = self.means_[i]
            self.bs_[i] = np.dot(np.eye(self.n_features) - self.As_[i], mu)

