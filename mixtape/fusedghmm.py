"""
`fusedgmhmm` implements a gaussian hidden Markov model with a pairwise L1 fusion
penality on the means of the output distributions.
"""
# Author: Robert McGibbon <rmcgibbo@gmail.com>
# Copyright (c) 2013, Stanford University
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without modification,
# are permitted provided that the following conditions are met:
#
#   Redistributions of source code must retain the above copyright notice, this
#   list of conditions and the following disclaimer.
#
#   Redistributions in binary form must reproduce the above copyright notice, this
#   list of conditions and the following disclaimer in the documentation and/or
#   other materials provided with the distribution.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
# ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
# WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR
# ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
# (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
# LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON
# ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
# SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

#-----------------------------------------------------------------------------
# Imports
#-----------------------------------------------------------------------------

from __future__ import print_function, division

import string
import numpy as np
from sklearn import cluster
from sklearn.mixture import sample_gaussian, log_multivariate_normal_density
from mixtape.basehmm import _ReversibleHMM

#-----------------------------------------------------------------------------
# Code
#-----------------------------------------------------------------------------


class GaussianFusionHMM(_ReversibleHMM):
    """
    Reversible Gaussian Hidden Markov Model L1-Fusion Regularization

    Notes
    -----
    """
    def __init__(self, n_components=1, n_em_iter=100, n_lqa_iter=10,
                 fusion_prior=1e-2, thresh=1e-2, reversible_type='mle',
                 transmat=None, transmat_prior=None, vars_prior=None,
                 vars_weight=1, params=string.ascii_letters,
                 random_state=None, init_params=string.ascii_letters):
        self.fusion_prior = fusion_prior
        self.vars_prior = vars_prior
        self.vars_weight = vars_weight
        self.n_lqa_iter = n_lqa_iter
        super(GaussianFusionHMM, self).__init__(
            n_components=n_components, n_iter=n_em_iter, thresh=thresh, reversible_type=reversible_type,
            transmat=transmat, transmat_prior=transmat_prior, params=params, random_state=random_state,
            init_params=init_params)

    def _generate_sample_from_state(self, state, random_state=None):
        return sample_gaussian(self._means_[state], self._vars_[state],
                               'diag', random_state=random_state)

    def _compute_log_likelihood(self, obs):
        return log_multivariate_normal_density(
            obs, self._means_, self._vars_, 'diag')

    def _init(self, obs, params):
        super(GaussianFusionHMM, self)._init(obs, params=params)

        if obs[0].ndim == 1:
            raise ValueError('Each observation sequence must be 2 dimensional')

        if (hasattr(self, 'n_features')
                and self.n_features != obs[0].shape[1]):
            raise ValueError('Unexpected number of dimensions, got %s but '
                             'expected %s' % (obs[0].shape[1],
                                              self.n_features))
        self.n_features = obs[0].shape[1]

        if 'm' in params:
            self._means_ = cluster.KMeans(
                n_clusters=self.n_components).fit(obs[0]).cluster_centers_
        if 'v' in params:
            self._vars_ = np.vstack([np.var(obs[0], axis=0)]* self.n_components)

    def _initialize_sufficient_statistics(self):
        stats = super(GaussianFusionHMM, self)._initialize_sufficient_statistics()
        stats['nobs'] = 0
        stats['post'] = np.zeros(self.n_components)
        stats['obs'] = np.zeros((self.n_components, self.n_features))
        stats['obs**2'] = np.zeros((self.n_components, self.n_features))
        return stats

    def _accumulate_sufficient_statistics(self, stats, obs, framelogprob,
                                          posteriors, fwdlattice, bwdlattice,
                                          params):
        super(GaussianFusionHMM, self)._accumulate_sufficient_statistics(
            stats, obs, framelogprob, posteriors, fwdlattice, bwdlattice,
            params)

        stats['nobs'] += len(obs)
        if 'm' in params or 'v' in params:
            stats['post'] += posteriors.sum(axis=0)
            stats['obs'] += np.dot(posteriors.T, obs)

        if 'v' in params:
            stats['obs**2'] += np.dot(posteriors.T, obs ** 2)


    def _do_mstep(self, stats, params):
        super(GaussianFusionHMM, self)._do_mstep(stats, params)
        difference_cutoff = 1e-10

        denom = stats['post'][:, np.newaxis]
        def getdiff(means):
            diff = np.zeros((self.n_features, self.n_components, self.n_components))
            for i in range(self.n_features):
                diff[i] = np.maximum(np.abs(np.subtract.outer(means[:, i], means[:, i])), difference_cutoff)
            return diff

        if 'm' in params:
            means = stats['obs'] / denom  # unregularized means
            strength = self.fusion_prior / getdiff(means)  # adaptive regularization strength
            rhs =  stats['obs'] / self._vars_
            for i in range(self.n_features):
                np.fill_diagonal(strength[i], 0)

            for s in range(self.n_lqa_iter):
                diff = getdiff(means)
                if np.all(diff <= difference_cutoff):
                    break
                offdiagonal = -strength / diff
                diagonal_penalty = np.sum(strength/diff, axis=2)
                for f in range(self.n_features):
                    if np.all(diff[f] <= difference_cutoff):
                        continue
                    ridge_approximation = np.diag(stats['post'] / self._vars_[:, f] + diagonal_penalty[f]) + offdiagonal[f]
                    means[:, f] = np.linalg.solve(ridge_approximation, rhs[:, f])

            for i in range(self.n_features):
                for k, j in zip(*np.triu_indices(self.n_components)):
                    if diff[i, k, j] <= difference_cutoff:
                        means[k, i] = means[j, i]
            self._means_ = means

        if 'v' in params:
            vars_prior = self.vars_prior
            vars_weight = self.vars_weight
            if vars_prior is None:
                vars_weight = 0
                vars_prior = 0

            var_num = (stats['obs**2']
                       - 2 * self._means_ * stats['obs']
                       + self._means_ ** 2 * denom)
            var_denom = max(vars_weight - 1, 0) + denom
            self._vars_ = (vars_prior + var_num) / var_denom
