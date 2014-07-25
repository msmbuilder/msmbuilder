"""
`ghmm` implements a gaussian hidden Markov model with an optional
 pairwise L1 fusion penality on the means of the output distributions.
"""
# Author: Robert McGibbon <rmcgibbo@gmail.com>
# Contributors:
# Copyright (c) 2013, Stanford University
# All rights reserved.

# Mixtape is free software: you can redistribute it and/or modify
# it under the terms of the GNU Lesser General Public License as
# published by the Free Software Foundation, either version 2.1
# of the License, or (at your option) any later version.
#
# This library is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Lesser General Public License for more details.
#
# You should have received a copy of the GNU Lesser General Public
# License along with Mixtape. If not, see <http://www.gnu.org/licenses/>.

#-----------------------------------------------------------------------------
# Imports
#-----------------------------------------------------------------------------

from __future__ import print_function, division, absolute_import

import time
import warnings
import numpy as np
import random
from mixtape.discrete_approx import discrete_approx_mvn, NotSatisfiableError
from sklearn import cluster
import sklearn.mixture
_AVAILABLE_PLATFORMS = ['cpu', 'sklearn']
from mixtape import _ghmm, _reversibility
from mdtraj.utils import ensure_type
from sklearn.utils import check_random_state

try:
    from mixtape import _cuda_ghmm_single
    from mixtape import _cuda_ghmm_mixed
    _AVAILABLE_PLATFORMS.append('cuda')
except ImportError:
    pass

EPS = np.finfo(np.float32).eps

#-----------------------------------------------------------------------------
# Code
#-----------------------------------------------------------------------------


class GaussianFusionHMM(object):

    """
    Reversible Gaussian Hidden Markov Model L1-Fusion Regularization

    Parameters
    ----------
    n_states : int
        The number of components (states) in the model
    n_init : int
        Number of time the EM algorithm will be run with different
        random seeds. The final results will be the best output of
        n_init consecutive runs in terms of log likelihood.
    n_em_iter : int
        The maximum number of iterations of expectation-maximization to
        run during each fitting round.
    n_lqa_iter : int
        The number of iterations of the local quadratic approximation fixed
        point equations to solve when computing the new means with a nonzero
        L1 fusion penalty.
    thresh : float
        Convergence threshold for the log-likelihood during expectation
        maximization. When the increase in the log-likelihood is less
        than thresh between subsequent rounds of E-M, fitting will finish.
    fusion_prior : float
        The strength of the L1 fusion prior.
    reversible_type : str
        Method by which the reversibility of the transition matrix
        is enforced. 'mle' uses a maximum likelihood method that is
        solved by numerical optimization (BFGS), and 'transpose'
        uses a more restrictive (but less computationally complex)
        direct symmetrization of the expected number of counts.
    transmat_prior : float, optiibal
        A prior on the transition matrix entries. If supplied, a
        psuedocount of transmat_prior - 1 is added to each entry
        in the expected number of observed transitions from each state
        to each other state, so this is like a uniform dirichlet alpha
        in a sense.
    vars_prior : float, optional
        A prior used on the variance. This can be useful in the undersampled
        regime where states may be collapsing onto a single point, but
        is generally not needed.
    vars_weight : float, optional
        Weight of the vars prior
    random_states : int, optional
        Random state, used during sampling.
    params : str
        A string with the parameters to optimizing during the fitting.
        If 't' is in params, the transition matrix will be optimized. If
        'm' is in params, the statemeans will be optimized. If 'v' is in
        params, the state variances will be optimized.
    init_params : str
        A string with the parameters to initialize prior to fitting.
        If 't' is in params, the transition matrix will be set. If
        'm' is in params, the statemeans will be set. If 'v' is in
        params, the state variances will be set.
    timing : bool, default=False
        Print detailed timing information about the fitting process.
    n_hotstart : {int, 'all'}
        Number of sequences to use when hotstarting the EM with kmeans.
        Default='all'
    init_algo : str
        Use this algorithm to hotstart the means and covariances.  Must
        be one of "kmeans" or "GMM"

    Attributes
    ----------
    means_ :
    vars_ :
    transmat_ :
    populations_ :
    fit_logprob_ :

    Notes
    -----
    """

    def __init__(self, n_states, n_features, n_init=10, n_em_iter=10,
                 n_lqa_iter=10, fusion_prior=1e-2, thresh=1e-2,
                 reversible_type='mle', transmat_prior=None, vars_prior=1e-3,
                 vars_weight=1, random_state=None, params='tmv',
                 init_params='tmv', platform='cpu', precision='mixed',
                 timing=False, n_hotstart='all', init_algo="kmeans"):
        self.n_states = n_states
        self.n_init = n_init
        self.n_features = n_features
        self.n_em_iter = n_em_iter
        self.n_lqa_iter = n_lqa_iter
        self.fusion_prior = fusion_prior
        self.thresh = thresh
        self.reversible_type = reversible_type
        self.transmat_prior = transmat_prior
        self.vars_prior = vars_prior
        self.vars_weight = vars_weight
        self.random_state = random_state
        self.params = params
        self.init_params = init_params
        self.platform = platform
        self.timing = timing
        self.n_hotstart = n_hotstart
        self.init_algo = init_algo
        self._impl = None

        if not reversible_type in ['mle', 'transpose']:
            raise ValueError('Invalid value for reversible_type: %s '
                             'Must be either "mle" or "transpose"'
                             % reversible_type)
        if n_init < 1:
            raise ValueError('HMM estimation requires at least one run')
        if n_em_iter < 1:
            raise ValueError('HMM estimation requires at least one em iter')

        if self.platform == 'cpu':
            self._impl = _ghmm.GaussianHMMCPUImpl(
                self.n_states, self.n_features, precision)
        elif self.platform == 'sklearn':
            self._impl = _SklearnGaussianHMMCPUImpl(self.n_states, self.n_features)
        elif self.platform == 'cuda':
            if precision == 'single':
                self._impl = _cuda_ghmm_single.GaussianHMMCUDAImpl(
                    self.n_states, self.n_features)
            elif precision == 'mixed':
                self._impl = _cuda_ghmm_mixed.GaussianHMMCUDAImpl(
                    self.n_states, self.n_features)
            else:
                raise ValueError('Only single and mixed precision are supported on CUDA')
        else:
            raise ValueError('Invalid platform "%s". Available platforms are '
                             '%s.' % (platform, ', '.join(_AVAILABLE_PLATFORMS)))
        
        if self.transmat_prior is None:
            self.transmat_prior = 1.0

        if self.init_algo not in ["GMM", "kmeans"]:
            raise ValueError("init_algo must be either GMM or kmeans")

    def fit(self, sequences, y=None):
        """Estimate model parameters.

        An initialization step is performed before entering the EM
        algorithm. If you want to avoid this step, pass proper
        ``init_params`` keyword argument to estimator's constructor.

        Parameters
        ----------
        sequences : list
            List of 2-dimensional array observation sequences, each of which
            has shape (n_samples_i, n_features), where n_samples_i
            is the length of the i_th observation.
        y : unused
            Needed for sklearn API consistency.
        """
        n_obs = sum(len(s) for s in sequences)
        best_fit = {'params': {}, 'loglikelihood': -np.inf}
        # counter for the total number of EM iters performed
        total_em_iters = 0
        if self.timing:
            start_time = time.time()

        for _ in range(self.n_init):
            fit_logprob = []
            self._init(sequences, self.init_params)
            for i in range(self.n_em_iter):
                # Expectation step
                curr_logprob, stats = self._impl.do_estep()
                if stats['trans'].sum() > 10 * n_obs:
                    raise OverflowError((
                        'Number of transition counts: %s. Total sequence length = %s '
                        'Numerical overflow detected. Try splitting your trajectories '
                        'into shorter segments or running in double ' % (
                            stats['trans'].sum(), n_obs)))

                fit_logprob.append(curr_logprob)
                # Check for convergence
                if i > 0 and abs(fit_logprob[-1] - fit_logprob[-2]) < self.thresh:
                    break

                # Maximization step
                self._do_mstep(stats, self.params)
                total_em_iters += 1

            # if this is better than our other iterations, keep it
            if curr_logprob > best_fit['loglikelihood']:
                best_fit['loglikelihood'] = curr_logprob
                best_fit['params'] = {'means': self.means_,
                                      'vars': self.vars_,
                                      'populations': self.populations_,
                                      'transmat': self.transmat_,
                                      'fit_logprob': fit_logprob}

        # Set the final values
        self.means_ = best_fit['params']['means']
        self.vars_ = best_fit['params']['vars']
        self.transmat_ = best_fit['params']['transmat']
        self.populations_ = best_fit['params']['populations']
        self.fit_logprob_ = best_fit['params']['fit_logprob']

        if self.timing:
            # but only print the timing variables if people really want them
            s_per_sample_per_em = (time.time() - start_time) / \
                (sum(len(s) for s in sequences) * total_em_iters)
            print('GaussianFusionHMM EM Fitting')
            print('----------------------------')
            print('Platform: %s    n_features: %d' % (self.platform, self.n_features))
            print('TOTAL EM Iters: %s' % total_em_iters)
            print('Speed:    %.3f +/- %.3f us/(sample * em-iter)' % (
                np.mean(s_per_sample_per_em * 10 ** 6),
                np.std(s_per_sample_per_em * 10 ** 6)))

        return self

    def _init(self, sequences, init_params):
        '''
        Find initial means(hot start)
        '''
        sequences = [ensure_type(s, dtype=np.float32, ndim=2, name='s', warn_on_cast=False)
                     for s in sequences]
        self._impl._sequences = sequences

        if self.n_hotstart == 'all':
            small_dataset = np.vstack(sequences)
        else:
            small_dataset = np.vstack(sequences[0:min(len(sequences), self.n_hotstart)])
        
        if self.init_algo == "GMM" and ("m" in init_params or "v" in init_params):
            mixture = sklearn.mixture.GMM(self.n_states, n_init=1, random_state=self.random_state)
            mixture.fit(small_dataset)
            if "m" in init_params:
                self.means_ = mixture.means_
            if "v" in init_params:
                self.vars_ = mixture.covars_
        else:
            if 'm' in init_params:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    self.means_ = cluster.KMeans(
                        n_clusters=self.n_states, n_init=1, init='random',
                        n_jobs=-1, random_state=self.random_state).fit(
                        small_dataset).cluster_centers_
            if 'v' in init_params:
                self.vars_ = np.vstack([np.var(small_dataset, axis=0)] * self.n_states)
        if 't' in init_params:
            transmat_ = np.empty((self.n_states, self.n_states))
            transmat_.fill(1.0 / self.n_states)
            self.transmat_ = transmat_
            self.populations_ = np.ones(self.n_states) / self.n_states

    def _do_mstep(self, stats, params):
        if 't' in params:
            if self.reversible_type == 'mle':
                counts = np.maximum(
                    stats['trans'] + self.transmat_prior - 1.0, 1e-20).astype(np.float64)
                self.transmat_, self.populations_ = _reversibility.reversible_transmat(
                    counts)
            elif self.reversible_type == 'transpose':
                revcounts = np.maximum(
                    self.transmat_prior - 1.0 + stats['trans'] + stats['trans'].T, 1e-20)
                populations = np.sum(revcounts, axis=0)
                self.populations_ = populations / np.sum(populations)
                self.transmat_ = revcounts / np.sum(revcounts, axis=1)[:, np.newaxis]
            else:
                raise ValueError('Invalid value for reversible_type: %s '
                                 'Must be either "mle" or "transpose"'
                                 % self.reversible_type)

        difference_cutoff = 1e-10
        # we don't want denom to be zero, because then the new value of the means
        # will be nan/inf. so padd it up by a very small constant. This particular
        # padding is following the sklearn mixture model m_step code from
        # https://github.com/scikit-learn/scikit-learn/blob/master/sklearn/mixture/gmm.py#L496
        denom = (stats['post'][:, np.newaxis] + 10 * EPS)

        def getdiff(means):
            diff = np.zeros((self.n_features, self.n_states, self.n_states))
            for i in range(self.n_features):
                diff[i] = np.maximum(
                    np.abs(np.subtract.outer(means[:, i], means[:, i])), difference_cutoff)
            return diff

        if 'm' in params:
            means = stats['obs'] / denom  # unregularized means

            if self.fusion_prior > 0 and self.n_lqa_iter > 0:
                # adaptive regularization strength
                strength = self.fusion_prior / getdiff(means)
                rhs = stats['obs'] / self.vars_
                for i in range(self.n_features):
                    np.fill_diagonal(strength[i], 0)

                break_lqa = False
                for s in range(self.n_lqa_iter):
                    diff = getdiff(means)
                    if np.all(diff <= difference_cutoff) or break_lqa:
                        break

                    offdiagonal = -strength / diff
                    diagonal_penalty = np.sum(strength / diff, axis=2)
                    for f in range(self.n_features):
                        if np.all(diff[f] <= difference_cutoff):
                            continue
                        ridge_approximation = np.diag(
                            stats['post'] / self.vars_[:, f] + diagonal_penalty[f]) + offdiagonal[f]
                        try:
                            means[:, f] = np.linalg.solve(ridge_approximation, rhs[:, f])
                        except np.linalg.LinAlgError:
                            # I'm not really sure what exactly causes the ridge
                            # approximation to be non-solvable, but it probably
                            # means we're too close to the merging. Maybe 1e-10
                            # is cutting it too close. ANyways, just break now and
                            # use the last valid value of the means.
                            break_lqa = True

                for i in range(self.n_features):
                    for k, j in zip(*np.triu_indices(self.n_states)):
                        if diff[i, k, j] <= difference_cutoff:
                            means[k, i] = means[j, i]

            self.means_ = means

        if 'v' in params:
            vars_prior = self.vars_prior
            vars_weight = self.vars_weight
            if vars_prior is None:
                vars_weight = 0
                vars_prior = 0

            var_num = (stats['obs**2']
                       - 2 * self.means_ * stats['obs']
                       + self.means_ ** 2 * denom)
            var_denom = max(vars_weight - 1, 0) + denom
            self.vars_ = (vars_prior + var_num) / var_denom

    @property
    def means_(self):
        return self._means_

    @means_.setter
    def means_(self, value):
        value = np.asarray(value, order='c', dtype=np.float32)
        self._means_ = value
        self._impl.means_ = value

    @property
    def vars_(self):
        return self._vars_

    @vars_.setter
    def vars_(self, value):
        value = np.asarray(value, order='c', dtype=np.float32)
        self._vars_ = value
        self._impl.vars_ = value

    @property
    def transmat_(self):
        return self._transmat_

    @transmat_.setter
    def transmat_(self, value):
        value = np.asarray(value, order='c', dtype=np.float32)
        self._transmat_ = value
        self._impl.transmat_ = value

    @property
    def populations_(self):
        return self._populations_

    @populations_.setter
    def populations_(self, value):
        value = np.asarray(value, order='c', dtype=np.float32)
        self._populations_ = value
        self._impl.startprob_ = value

    @property
    def timescales_(self):
        """The implied relaxation timescales of the hidden Markov transition
        matrix

        By diagonalizing the transition matrix, its propagation of an arbitrary
        initial probability vector can be written as a sum of the eigenvectors
        of the transition weighted by per-eigenvector term that decays
        exponentially with time. Each of these eigenvectors describes a
        "dynamical mode" of the transition matrix and has a characteristic
        timescale, which gives the timescale on which that mode decays towards
        equilibrium. These timescales are given by :math:`-1/log(u_i)` where
        :math:`u_i` are the eigenvalues of the transition matrix. In a
        reversible HMM with N states, the number of timescales is at most N-1.
        (The -1 comes from the fact that the stationary distribution of the chain
        is associated with an eigenvalue of 1, and an infinite characteristic
        timescale). The number of timescales can be less than N-1 for every
        eigenvalue of the transition matrix that is negative (which is
        allowable by detailed balance).

        Returns
        -------
        timescales : array, shape=[n_timescales]
            The characteristic timescales of the transition matrix. If the model
            has not been fit or does not have a transition matrix, the return
            value will be None.
        """
        if self.transmat_ is None:
            return None
        try:
            eigvals = np.linalg.eigvals(self.transmat_)

            # sort the eigenvalues in descending order (i.e. sort + reverse),
            # then discard the very first (largest) eigenvalue, which is the
            # stationary one.
            eigvals = np.sort(eigvals)[::-1][1:]

            # retain only eigenvalues > 0. These are the ones that correspond
            # to implied timescales. Eigenvalues below zero are possible (the
            # detailed balance constraint guarentees that the eigenvalues are
            # in -1 < l <= 1, but not that they strictly positive). But
            # eigevalues below zero do not have a real interpretation as
            # "implied timescales" because they correspond to sort of
            # damped oscillatory decays.

            eigvals = eigvals[eigvals > 0]
            return -1.0 / np.log(eigvals)
        except np.linalg.LinAlgError:
            # this can happen if the transition matrix contains Nans
            # or Infs, and possibly for other reasons like convergence
            return np.nan * np.ones(self.n_states - 1)

    def score(self, sequences):
        """Log-likelihood of sequences under the model

        Parameters
        ----------
        sequences : list
            List of 2-dimensional array observation sequences, each of which
            has shape (n_samples_i, n_features), where n_samples_i
            is the length of the i_th observation.
        """
        sequences = [ensure_type(s, dtype=np.float32, ndim=2, name='s')
                     for s in sequences]
        self._impl._sequences = sequences
        logprob, _ = self._impl.do_estep()
        return logprob

    def predict(self, sequences):
        """Find most likely hidden-state sequence corresponding to
        each data timeseries.

        Uses the Viterbi algorithm.

        Parameters
        ----------
        sequences : list
            List of 2-dimensional array observation sequences, each of which
            has shape (n_samples_i, n_features), where n_samples_i
            is the length of the i_th observation.

        Returns
        -------
        viterbi_logprob : float
            Log probability of the maximum likelihood path through the HMM.

        hidden_sequences : list of np.ndarrays[dtype=int, shape=n_samples_i]
            Index of the most likely states for each observation.
        """
        if not hasattr(self._impl, 'do_viterbi'):
            raise NotImplementedError(
                'The %s platform does not support this algorithm (yet)' %
                self.platform)

        self._impl._sequences = sequences
        logprob, state_sequences = self._impl.do_viterbi()

        return logprob, state_sequences


    def draw_centroids(self, sequences):
        """Find conformations most representative of model means.

        Parameters
        ----------
        sequences : list
            List of 2-dimensional array observation sequences, each of which
            has shape (n_samples_i, n_features), where n_samples_i
            is the length of the i_th observation.

        Returns
        -------
        centroid_pairs_by_state : np.ndarray, dtype=int, shape = (n_states, 1, 2)
            centroid_pairs_by_state[state, 0] = (trj, frame) gives the 
            trajectory and frame index associated with the 
            mean of `state`
        mean_approx : np.ndarray, dtype=float, shape = (n_states, 1, n_features)
            mean_approx[state, 0] gives the features at the representative 
            point for `state`

        See Also
        --------
        utils.map_drawn_samples : Extract conformations from MD trajectories by index.
        GaussianFusionHMM.draw_samples : Draw samples from GHMM
        

        """    
        
        logprob = [sklearn.mixture.log_multivariate_normal_density(x, self.means_, self.vars_, covariance_type='diag') for x in sequences]

        argm = np.array([lp.argmax(0) for lp in logprob])
        probm = np.array([lp.max(0) for lp in logprob])

        trj_ind = probm.argmax(0)
        frame_ind = argm[trj_ind, np.arange(self.n_states)]

        mean_approx = np.array([sequences[trj_ind_i][frame_ind_i] for trj_ind_i, frame_ind_i in zip(trj_ind, frame_ind)])
        
        centroid_pairs_by_state = np.array(list(zip(trj_ind, frame_ind)))
        
        return centroid_pairs_by_state[:, np.newaxis, :], mean_approx[:, np.newaxis, :]  # np.newaxis changes arrays from 2D to 3D for consistency with `sample_states()`


    def draw_samples(self, sequences, n_samples, scheme="even", match_vars=False):
        """Sample conformations from each state.

        Parameters
        ----------
        sequences : list
            List of 2-dimensional array observation sequences, each of which
            has shape (n_samples_i, n_features), where n_samples_i
            is the length of the i_th observation.
        n_samples : int
            How many samples to return from each state
        scheme : str, optional, default='even'
            Must be one of ['even', "maxent"].  
        match_vars : bool, default=False
            Flag for matching variances in maxent discrete approximation

        Returns
        -------
        selected_pairs_by_state : np.array, dtype=int, shape=(n_states, n_samples, 2)
            selected_pairs_by_state[state] gives an array of randomly selected (trj, frame)
            pairs from the specified state.
        sample_features : np.ndarray, dtype=float, shape = (n_states, n_samples, n_features)
            sample_features[state, sample] gives the features for the given `sample` of 
            `state`
            
        Notes
        -----
        With scheme='even', this function assigns frames to states crisply then samples from
        the uniform distribution on the frames belonging to each state.
        With scheme='maxent', this scheme uses a maximum entropy method to
        determine a discrete distribution on samples whose mean (and possibly variance)
        matches the GHMM means.

        See Also
        --------
        utils.map_drawn_samples : Extract conformations from MD trajectories by index.
        GaussianFusionHMM.draw_centroids : Draw centers from GHMM
        
        ToDo
        ----
        This function could be separated into several MixIns for
        models with crisp and fuzzy state assignments.  Then we could have
        an optional argument that specifies which way to do the sampling
        from states--e.g. use either the base class function or a 
        different one.
        """
        
        random = check_random_state(self.random_state)
        
        if scheme == 'even':
            logprob = [sklearn.mixture.log_multivariate_normal_density(x, self.means_, self.vars_, covariance_type='diag') for x in sequences]
            ass = [lp.argmax(1) for lp in logprob]
            
            selected_pairs_by_state = []
            for state in range(self.n_states):
                all_frames = [np.where(a == state)[0] for a in ass]
                pairs = [(trj, frame) for (trj, frames) in enumerate(all_frames) for frame in frames]
                selected_pairs_by_state.append([pairs[random.choice(len(pairs))] for i in range(n_samples)])
        
        elif scheme == "maxent":
            X_concat = np.concatenate(sequences)
            all_pairs = np.array([(trj, frame) for trj, X in enumerate(sequences) for frame in range(X.shape[0])])
            selected_pairs_by_state = []
            for k in range(self.n_states):
                print('computing weights for k=%d...' % k)
                try:
                    weights = discrete_approx_mvn(X_concat, self.means_[k], self.vars_[k], match_vars)
                except NotSatisfiableError:
                    self.error('Satisfiability failure. Could not match the means & '
                               'variances w/ discrete distribution. Try removing the '
                               'constraint on the variances with --no-match-vars?')

                weights /= weights.sum()
                frames = random.choice(len(all_pairs), n_samples, p=weights)
                selected_pairs_by_state.append(all_pairs[frames])

        else:
            raise(ValueError("scheme must be one of ['even', 'maxent'])"))
        
        return np.array(selected_pairs_by_state)

        
class _SklearnGaussianHMMCPUImpl(object):

    def __init__(self, n_states, n_features):
        from sklearn.hmm import GaussianHMM
        self.impl = GaussianHMM(n_states, params='stmc')

        self._sequences = None
        self.means_ = None
        self.vars_ = None
        self.transmat_ = None
        self.startprob_ = None

    def do_estep(self):
        from sklearn.utils.extmath import logsumexp

        self.impl.means_ = self.means_.astype(np.double)
        self.impl.covars_ = self.vars_.astype(np.double)
        self.impl.transmat_ = self.transmat_.astype(np.double)
        self.impl.startprob_ = self.startprob_.astype(np.double)
        stats = self.impl._initialize_sufficient_statistics()
        curr_logprob = 0
        for seq in self._sequences:
            seq = seq.astype(np.double)
            framelogprob = self.impl._compute_log_likelihood(seq)
            lpr, fwdlattice = self.impl._do_forward_pass(framelogprob)
            bwdlattice = self.impl._do_backward_pass(framelogprob)
            gamma = fwdlattice + bwdlattice
            posteriors = np.exp(gamma.T - logsumexp(gamma, axis=1)).T
            curr_logprob += lpr
            self.impl._accumulate_sufficient_statistics(
                stats, seq, framelogprob, posteriors, fwdlattice,
                bwdlattice, self.impl.params)

        return curr_logprob, stats

    def do_viterbi(self):
        logprob = 0
        state_sequences = []
        for obs in self._sequences:
            lpr, ss = self.impl._decode_viterbi(obs)
            logprob += lpr
            state_sequences.append(ss)

        return logprob, state_sequences
