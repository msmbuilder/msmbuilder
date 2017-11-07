import numpy as np
cimport numpy as np
from libcpp.vector cimport vector
from libcpp.string cimport string
from cpython.ref cimport PyObject

import time
import warnings
from sklearn import cluster, mixture
from sklearn.utils import check_random_state
from mdtraj.utils import ensure_type
from .discrete_approx import discrete_approx_mvn, NotSatisfiableError
from ..utils import check_iter_of_sequences, printoptions
from ..msm._markovstatemodel import _transmat_mle_prinz


cdef extern from "Trajectory.h" namespace "msmbuilder":
    cdef cppclass Trajectory:
        Trajectory(PyObject*, char*, int, int, int, int) except +
        Trajectory()

cdef extern from "GaussianHMMFitter.h" namespace "msmbuilder":
    cdef cppclass GaussianHMMFitter[T]:
        GaussianHMMFitter(GaussianHMM, int, int, int, double*) except +
        void set_transmat(double*)
        void set_means_and_variances(double*, double*)
        void fit(const vector[Trajectory]&, double)
        double score_trajectories(vector[Trajectory]&)
        double predict_state_sequence(Trajectory& trajectory, int* state_sequence)
        int get_fit_iterations()
        void get_transition_counts(double*)
        void get_obs(double*)
        void get_obs2(double*)
        void get_post(double*)
        void get_log_probability(double*)

cdef public class GaussianHMM[object GaussianHMMObject, type GaussianHMMType]:
    """Reversible Gaussian Hidden Markov Model L1-Fusion Regularization

    This model estimates Hidden Markov model for a vector dataset which is
    contained to be reversible (satisfy detailed balance) with Gaussian
    emission distributions. This model is similar to a ``MarkovStateModel``
    without a "hard" assignments of conformations to clusters. Optionally, it
    can apply L1-regularization to the positions of the Gaussians. See [1] for
    details.

    Parameters
    ----------
    n_states : int
        The number of components (states) in the model
    n_init : int
        Number of time the EM algorithm will be run with different
        random seeds. The final results will be the best output of
        n_init consecutive runs in terms of log likelihood.
    n_iter : int
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
    vars_prior : float, optional
        A prior used on the variance. This can be useful in the undersampled
        regime where states may be collapsing onto a single point, but
        is generally not needed.
    vars_weight : float, optional
        Weight of the vars prior
    random_state : int, optional
        Random state, used during sampling.
    timing : bool, default=False
        Print detailed timing information about the fitting process.
    n_hotstart : {int, 'all'}
        Number of sequences to use when hotstarting the EM.
        Default='all'
    init_algo : str
        Use this algorithm to hotstart the means and covariances.  Must
        be one of "kmeans" or "GMM"

    References
    ----------
    .. [1] McGibbon, Robert T. et al., "Understanding Protein Dynamics with
       L1-Regularized Reversible Hidden Markov Models" Proc. 31st Intl.
       Conf. on Machine Learning (ICML). 2014.

    Attributes
    ----------
    means_ :
    vars_ :
    transmat_ :
    populations_ :
    fit_logprob_ :
    """

    cdef int n_states, n_features, n_init, n_iter
    cdef float thresh
    cdef random_state, timing, n_hotstart
    cdef startprob
    cdef stats
    cdef reversible_type, n_lqa_iter, fusion_prior, vars_prior, vars_weight, init_algo
    cdef _means_, _vars_, _transmat_, _populations_, _fit_logprob_, _fit_time_

    def __init__(self, n_states, n_init=10, n_iter=10,
                 n_lqa_iter=10, fusion_prior=1e-2, thresh=1e-2,
                 reversible_type='mle', vars_prior=1e-3,
                 vars_weight=1, random_state=None,
                 timing=False, n_hotstart='all', init_algo='kmeans'):
        self.n_states = int(n_states)
        self.n_features = -1
        self.n_init = int(n_init)
        self.n_iter = int(n_iter)
        self.n_lqa_iter = int(n_lqa_iter)
        self.fusion_prior = float(fusion_prior)
        self.thresh = float(thresh)
        self.reversible_type = reversible_type
        self.vars_prior = float(vars_prior)
        self.vars_weight = float(vars_weight)
        self.random_state = random_state
        self.timing = timing
        self.n_hotstart = n_hotstart
        self.init_algo = init_algo
        self.startprob = np.tile(1.0/n_states, n_states)
        self.stats = {}

    @classmethod
    def _init_argspec(self):
        # Returns the equivalent of `inspect.getargspec(GaussianHMM.__init__)`.
        # this is required for the command line infastructure for all estimators
        # written in cython as a workaround, since cython modules don't
        # interact correctly with the `inspect` module.

        # any changes to the signature of __init__ need to be reflected here.
        from inspect import ArgSpec
        return ArgSpec(
        ['self', 'n_states', 'n_init', 'n_iter', 'n_lqa_iter',
         'fusion_prior', 'thresh', 'reversible_type', 'vars_prior',
         'vars_weight', 'random_state', 'timing',
         'n_hotstart', 'init_algo'],
          None, None,
          [10, 10, 10, 1e-2, 1e-2, 'mle', 1e-3, 1, None, False,
          'all', 'kmeans']
        )

    @property
    def means_(self):
        return self._means_

    @property
    def vars_(self):
        return self._vars_

    @property
    def transmat_(self):
        return self._transmat_

    @property
    def populations_(self):
        return self._populations_

    @property
    def fit_logprob_(self):
        return self._fit_logprob_

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
        if self._transmat_ is None:
            return None
        try:
            eigvals = np.linalg.eigvals(self._transmat_)

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
        GaussianHMM.draw_samples : Draw samples from GHMM
        """

        logprob = [mixture.log_multivariate_normal_density(
            x, self._means_, self._vars_, covariance_type='diag'
        ) for x in sequences]

        argm = np.array([lp.argmax(0) for lp in logprob])
        probm = np.array([lp.max(0) for lp in logprob])

        trj_ind = probm.argmax(0)
        frame_ind = argm[trj_ind, np.arange(self.n_states)]

        mean_approx = np.array([sequences[trj_ind_i][frame_ind_i]
                                for trj_ind_i, frame_ind_i
                                in zip(trj_ind, frame_ind)])

        centroid_pairs_by_state = np.array(list(zip(trj_ind, frame_ind)))

        # Change from 2D to 3D for consistency with `sample_states()`
        return (centroid_pairs_by_state[:, np.newaxis, :],
                mean_approx[:, np.newaxis, :])


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
        With scheme='even', this function assigns frames to states crisply
        then samples from the uniform distribution on the frames belonging
        to each state. With scheme='maxent', this scheme uses a maximum
        entropy method to determine a discrete distribution on samples
        whose mean (and possibly variance) matches the GHMM means.

        See Also
        --------
        utils.map_drawn_samples : Extract conformations from MD trajectories by index.
        GaussianHMM.draw_centroids : Draw centers from GHMM

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
            logprob = [
                mixture.log_multivariate_normal_density(
                    x, self._means_, self._vars_, covariance_type='diag'
                ) for x in sequences]
            ass = [lp.argmax(1) for lp in logprob]

            selected_pairs_by_state = []
            for state in range(self.n_states):
                all_frames = [np.where(a == state)[0] for a in ass]
                pairs = [(trj, frame) for (trj, frames)
                         in enumerate(all_frames) for frame in frames]
                selected_pairs_by_state.append(
                    [pairs[random.choice(len(pairs))]
                     for i in range(n_samples)]
                )

        elif scheme == "maxent":
            X_concat = np.concatenate(sequences)
            all_pairs = np.array([(trj, frame) for trj, X
                                  in enumerate(sequences)
                                  for frame in range(X.shape[0])])
            selected_pairs_by_state = []
            for k in range(self.n_states):
                try:
                    weights = discrete_approx_mvn(X_concat, self._means_[k],
                                                  self._vars_[k], match_vars)
                except NotSatisfiableError:
                    err = ''.join([
                        'Satisfiability failure. Could not match the means & ',
                        'variances w/ discrete distribution. Try removing the ',
                        'constraint on the variances with --no-match-vars?',
                    ])
                    self.error(err)

                weights /= weights.sum()
                frames = random.choice(len(all_pairs), n_samples, p=weights)
                selected_pairs_by_state.append(all_pairs[frames])

        else:
            raise(ValueError("scheme must be one of ['even', 'maxent'])"))

        return np.array(selected_pairs_by_state)

    def summarize(self):
        """Get a string summarizing the model."""
        with printoptions(precision=4, suppress=True):
            return """Gaussian HMM
------------
n_states : {n_states}
logprob: {logprob}
fit_time: {fit_time:.3f}s

populations: {populations}
transmat:
{transmat}
timescales: {timescales}
    """.format(n_states=self.n_states, logprob=self._fit_logprob_[-1],
               populations=str(self._populations_), transmat=str(self._transmat_),
               timescales=self.timescales_, fit_time=self._fit_time_)

    def fit_predict(self, sequences, y=None):
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
        self.fit(sequences, y=y)
        return self.predict(sequences)

    def fit(self, sequences, y=None):
        """Estimate model parameters.

        Parameters
        ----------
        sequences : list
            List of 2-dimensional array observation sequences, each of which
            has shape (n_samples_i, n_features), where n_samples_i
            is the length of the i_th observation.
        """
        self._validate_sequences(sequences)
        self.n_features = sequences[0].shape[1]
        dtype = sequences[0].dtype
        best_fit = {'params': {}, 'loglikelihood': -np.inf}
        start_time = time.time()
        self.stats = {}
        total_iters = 0
        for run in range(self.n_init):
            self._init(sequences)
            if dtype == np.float32:
                self._fit_float(sequences)
            elif dtype == np.float64:
                self._fit_double(sequences)
            else:
                raise ValueError('Unsupported data type: '+str(dtype))
            total_iters += len(self.stats['log_probability'])

            # If this is better than our other runs, keep it
            if self.stats['log_probability'][-1] > best_fit['loglikelihood']:
                best_fit['loglikelihood'] = self.stats['log_probability'][-1]
                best_fit['params'] = {'means': self.means_,
                                      'vars': self.vars_,
                                      'populations': self.populations_,
                                      'transmat': self.transmat_,
                                      'fit_logprob': self.stats['log_probability']}

        # Set the final values
        self._means_ = best_fit['params']['means']
        self._vars_ = best_fit['params']['vars']
        self._transmat_ = best_fit['params']['transmat']
        self._populations_ = best_fit['params']['populations']
        self._fit_logprob_ = best_fit['params']['fit_logprob']
        self._fit_time_ = time.time() - start_time

        if self.timing:
            # Only print the timing variables if people really want them
            s_per_sample_per_em = (self._fit_time_) / (sum(len(s) for s in sequences) * total_iters)
            print('GaussianHMM EM Fitting')
            print('----------------------')
            print('n_features: %d' % (self.n_features))
            print('TOTAL EM Iters: %s' % total_iters)
            print('Speed:    %.3f +/- %.3f us/(sample * em-iter)' % (
                np.mean(s_per_sample_per_em * 10 ** 6),
                np.std(s_per_sample_per_em * 10 ** 6)))
        return self

    def _validate_sequences(self, sequences):
        """Make sure the sequences supplied by the user are valid."""
        if len(sequences) == 0:
           raise ValueError('sequences is empty')
        check_iter_of_sequences(sequences)
        dtype = sequences[0].dtype
        if any(s.dtype != dtype for s in sequences):
            raise ValueError('All sequences must have the same data type')
        if self.n_features == -1:
            # It hasn't been set yet.
            n_features = sequences[0].shape[1]
        else:
            n_features = self.n_features
        if any(s.shape[1] != n_features for s in sequences):
            raise ValueError('All sequences must have %d features' % n_features)

    cdef vector[Trajectory] _convert_sequences_to_vector_float(self, sequences):
        """Convert the sequences supplied by the user into the form needed by the C++ code."""
        cdef vector[Trajectory] trajectoryVec
        cdef np.ndarray[float, ndim=2] array
        for s in sequences:
            array = s
            trajectoryVec.push_back(Trajectory(<PyObject*> array, <char*> &array[0,0], array.shape[0], array.shape[1], array.strides[0], array.strides[1]))
        return trajectoryVec

    cdef vector[Trajectory] _convert_sequences_to_vector_double(self, sequences):
        """Convert the sequences supplied by the user into the form needed by the C++ code."""
        cdef vector[Trajectory] trajectoryVec
        cdef np.ndarray[double, ndim=2] array
        for s in sequences:
            array = s
            trajectoryVec.push_back(Trajectory(<PyObject*> array, <char*> &array[0,0], array.shape[0], array.shape[1], array.strides[0], array.strides[1]))
        return trajectoryVec

    def _init(self, sequences):
        """Find initial means (hot start)"""
        sequences = [ensure_type(s, dtype=np.float32, ndim=2, name='s', warn_on_cast=False)
                     for s in sequences]

        if self.n_hotstart == 'all':
            small_dataset = np.vstack(sequences)
        else:
            small_dataset = np.vstack(sequences[0:min(len(sequences), self.n_hotstart)])

        if self.init_algo == "GMM":
            mix = mixture.GMM(self.n_states, n_init=1, random_state=self.random_state)
            mix.fit(small_dataset)
            self._means_ = mix.means_
            self._vars_ = mix.covars_
        else:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                self._means_ = cluster.MiniBatchKMeans(
                    n_clusters=self.n_states, n_init=1, init='random',
                    random_state=self.random_state).fit(
                    small_dataset).cluster_centers_
            self._vars_ = np.vstack([np.var(small_dataset, axis=0)] * self.n_states)
        self._populations_ = np.ones(self.n_states) / self.n_states
        self._transmat_ = np.empty((self.n_states, self.n_states))
        self._transmat_.fill(1.0/self.n_states)

    def _fit_float(self, sequences):
        cdef vector[Trajectory] trajectoryVec
        cdef np.ndarray[double, ndim=1] startprob
        cdef np.ndarray[double, ndim=2] transmat
        cdef np.ndarray[double, ndim=2] means
        cdef np.ndarray[double, ndim=2] vars
        trajectoryVec = self._convert_sequences_to_vector_float(sequences)
        startprob = self.startprob
        transmat = self._transmat_
        means = self._means_.astype(np.float64)
        vars = self._vars_.astype(np.float64)
        cdef GaussianHMMFitter[float] *fitter = new GaussianHMMFitter[float](self, self.n_states, self.n_features, self.n_iter, <double*> &startprob[0])
        fitter.set_transmat(<double*> &transmat[0,0])
        fitter.set_means_and_variances(<double*> &means[0,0], <double*> &vars[0,0])
        try:
            fitter.fit(trajectoryVec, self.thresh)
        finally:
            del fitter

    def _fit_double(self, sequences):
        cdef vector[Trajectory] trajectoryVec
        cdef np.ndarray[double, ndim=1] startprob
        cdef np.ndarray[double, ndim=2] transmat
        cdef np.ndarray[double, ndim=2] means
        cdef np.ndarray[double, ndim=2] vars
        trajectoryVec = self._convert_sequences_to_vector_double(sequences)
        startprob = self.startprob
        transmat = self._transmat_
        means = self._means_.astype(np.float64)
        vars = self._vars_.astype(np.float64)
        cdef GaussianHMMFitter[double] *fitter = new GaussianHMMFitter[double](self, self.n_states, self.n_features, self.n_iter, <double*> &startprob[0])
        fitter.set_transmat(<double*> &transmat[0,0])
        fitter.set_means_and_variances(<double*> &means[0,0], <double*> &vars[0,0])
        try:
            fitter.fit(trajectoryVec, self.thresh)
        finally:
            del fitter

    def _getdiff(self, means, difference_cutoff):
        diff = np.zeros((self.n_features, self.n_states, self.n_states))
        for i in range(self.n_features):
            diff[i] = np.maximum(
                np.abs(np.subtract.outer(means[:, i], means[:, i])), difference_cutoff)
        return diff

    def _do_mstep(self):
        stats = self.stats
        transmat_prior = 1.0
        if self.reversible_type == 'mle':
            counts = np.maximum(
                np.nan_to_num(stats['trans']) + transmat_prior - 1.0,
                    1e-20).astype(np.float64)
            self._transmat_, self._populations_ = _transmat_mle_prinz(counts)
        elif self.reversible_type == 'transpose':
            revcounts = np.maximum(
                transmat_prior - 1.0 + stats['trans'] + stats['trans'].T, 1e-20)
            populations = np.sum(revcounts, axis=0)
            self._populations_ = populations / np.sum(populations)
            self._transmat_ = revcounts / np.sum(revcounts, axis=1)[:, np.newaxis]
        else:
            raise ValueError('Invalid value for reversible_type: %s '
                             'Must be either "mle" or "transpose"'
                             % self.reversible_type)

        difference_cutoff = 1e-10
        # we don't want denom to be zero, because then the new value of the means
        # will be nan/inf. so padd it up by a very small constant. This particular
        # padding is following the sklearn mixture model m_step code from
        # https://github.com/scikit-learn/scikit-learn/blob/master/sklearn/mixture/gmm.py#L496
        EPS = np.finfo(np.float32).eps
        denom = (stats['post'][:, np.newaxis] + 10 * EPS)

        means = stats['obs'] / denom  # unregularized means

        if self.fusion_prior > 0 and self.n_lqa_iter > 0:
            # adaptive regularization strength
            strength = self.fusion_prior / self._getdiff(means, difference_cutoff)
            rhs = stats['obs'] / self._vars_
            for i in range(self.n_features):
                np.fill_diagonal(strength[i], 0)

            break_lqa = False
            for s in range(self.n_lqa_iter):
                diff = self._getdiff(means, difference_cutoff)
                if np.all(diff <= difference_cutoff) or break_lqa:
                    break

                offdiagonal = -strength / diff
                diagonal_penalty = np.sum(strength / diff, axis=2)
                for f in range(self.n_features):
                    if np.all(diff[f] <= difference_cutoff):
                        continue
                    ridge_approximation = np.diag(
                        stats['post'] / self._vars_[:, f] + diagonal_penalty[f]) + offdiagonal[f]
                    try:
                        means[:, f] = np.linalg.solve(ridge_approximation, rhs[:, f])
                    except np.linalg.LinAlgError:
                        # I'm not really sure what exactly causes the ridge
                        # approximation to be non-solvable, but it probably
                        # means we're too close to the merging. Maybe 1e-10
                        # is cutting it too close. Anyways,
                        # just break now and use the last valid value
                        # of the means.
                        break_lqa = True

            for i in range(self.n_features):
                for k, j in zip(*np.triu_indices(self.n_states)):
                    if diff[i, k, j] <= difference_cutoff:
                        means[k, i] = means[j, i]

        self._means_ = means

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

    def score(self, sequences):
        """Log-likelihood of sequences under the model

        Parameters
        ----------
        sequences : list
            List of 2-dimensional array observation sequences, each of which
            has shape (n_samples_i, n_features), where n_samples_i
            is the length of the i_th observation.
        """
        self._validate_sequences(sequences)
        dtype = sequences[0].dtype
        if dtype == np.float32:
            return self._score_float(sequences)
        elif dtype == np.float64:
            return self._score_double(sequences)
        else:
            raise ValueError('Unsupported data type: '+str(dtype))

    cdef _score_float(self, sequences):
        cdef vector[Trajectory] trajectoryVec
        cdef np.ndarray[double, ndim=1] startprob
        cdef np.ndarray[double, ndim=2] transmat
        cdef np.ndarray[double, ndim=2] means
        cdef np.ndarray[double, ndim=2] vars
        trajectoryVec = self._convert_sequences_to_vector_float(sequences)
        startprob = self.startprob
        transmat = self._transmat_
        means = self._means_.astype(np.float64)
        vars = self._vars_.astype(np.float64)
        cdef GaussianHMMFitter[float] *fitter = new GaussianHMMFitter[float](self, self.n_states, self.n_features, self.n_iter, <double*> &startprob[0])
        fitter.set_transmat(<double*> &transmat[0,0])
        fitter.set_means_and_variances(<double*> &means[0,0], <double*> &vars[0,0])
        try:
            return fitter.score_trajectories(trajectoryVec)
        finally:
            del fitter

    cdef _score_double(self, sequences):
        cdef vector[Trajectory] trajectoryVec
        cdef np.ndarray[double, ndim=1] startprob
        cdef np.ndarray[double, ndim=2] transmat
        cdef np.ndarray[double, ndim=2] means
        cdef np.ndarray[double, ndim=2] vars
        trajectoryVec = self._convert_sequences_to_vector_double(sequences)
        startprob = self.startprob
        transmat = self._transmat_
        means = self._means_.astype(np.float64)
        vars = self._vars_.astype(np.float64)
        cdef GaussianHMMFitter[double] *fitter = new GaussianHMMFitter[double](self, self.n_states, self.n_features, self.n_iter, <double*> &startprob[0])
        fitter.set_transmat(<double*> &transmat[0,0])
        fitter.set_means_and_variances(<double*> &means[0,0], <double*> &vars[0,0])
        try:
            return fitter.score_trajectories(trajectoryVec)
        finally:
            del fitter

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
        self._validate_sequences(sequences)
        dtype = sequences[0].dtype
        if dtype == np.float32:
            return self._predict_float(sequences)
        elif dtype == np.float64:
            return self._predict_double(sequences)
        else:
            raise ValueError('Unsupported data type: '+str(dtype))

    cdef _predict_float(self, sequences):
        cdef Trajectory trajectory
        cdef np.ndarray[np.int32_t, ndim=1] state_sequence
        cdef np.ndarray[float, ndim=2] array
        cdef np.ndarray[double, ndim=1] startprob
        cdef np.ndarray[double, ndim=2] transmat
        cdef np.ndarray[double, ndim=2] means
        cdef np.ndarray[double, ndim=2] vars
        startprob = self.startprob
        transmat = self._transmat_
        means = self._means_.astype(np.float64)
        vars = self._vars_.astype(np.float64)
        cdef GaussianHMMFitter[float] *fitter = new GaussianHMMFitter[float](self, self.n_states, self.n_features, self.n_iter, <double*> &startprob[0])
        fitter.set_transmat(<double*> &transmat[0,0])
        fitter.set_means_and_variances(<double*> &means[0,0], <double*> &vars[0,0])
        try:
            logprob = 0.0
            viterbi_sequences = []
            for s in sequences:
                array = s
                trajectory = Trajectory(<PyObject*> array, <char*> &array[0,0], array.shape[0], array.shape[1], array.strides[0], array.strides[1])
                state_sequence = np.empty(len(s), dtype=np.int32)
                logprob += fitter.predict_state_sequence(trajectory, <int*> &state_sequence[0])
                viterbi_sequences.append(state_sequence)
            return (logprob, viterbi_sequences)
        finally:
            del fitter

    cdef _predict_double(self, sequences):
        cdef Trajectory trajectory
        cdef np.ndarray[np.int32_t, ndim=1] state_sequence
        cdef np.ndarray[double, ndim=2] array
        cdef np.ndarray[double, ndim=1] startprob
        cdef np.ndarray[double, ndim=2] transmat
        cdef np.ndarray[double, ndim=2] means
        cdef np.ndarray[double, ndim=2] vars
        startprob = self.startprob
        transmat = self._transmat_
        means = self._means_.astype(np.float64)
        vars = self._vars_.astype(np.float64)
        cdef GaussianHMMFitter[double] *fitter = new GaussianHMMFitter[double](self, self.n_states, self.n_features, self.n_iter, <double*> &startprob[0])
        fitter.set_transmat(<double*> &transmat[0,0])
        fitter.set_means_and_variances(<double*> &means[0,0], <double*> &vars[0,0])
        try:
            logprob = 0.0
            viterbi_sequences = []
            for s in sequences:
                array = s
                trajectory = Trajectory(<PyObject*> array, <char*> &array[0,0], array.shape[0], array.shape[1], array.strides[0], array.strides[1])
                state_sequence = np.empty(len(s), dtype=np.int32)
                logprob += fitter.predict_state_sequence(trajectory, <int*> &state_sequence[0])
                viterbi_sequences.append(state_sequence)
            return (logprob, viterbi_sequences)
        finally:
            del fitter

    cdef _record_stats_float(self, GaussianHMMFitter[float]* fitter):
        """Copy various statistics from the C++ class to this one."""
        cdef np.ndarray[double, ndim=2] transition_counts
        cdef np.ndarray[double, ndim=2] obs
        cdef np.ndarray[double, ndim=2] obs2
        cdef np.ndarray[double, ndim=1] post
        cdef np.ndarray[double, ndim=1] log_probability
        transition_counts = np.empty((self.n_states, self.n_states))
        obs = np.empty((self.n_states, self.n_features))
        obs2 = np.empty((self.n_states, self.n_features))
        post = np.empty(self.n_states)
        log_probability = np.empty(fitter.get_fit_iterations())
        fitter.get_transition_counts(<double*> &transition_counts[0,0])
        fitter.get_obs(<double*> &obs[0,0])
        fitter.get_obs2(<double*> &obs2[0,0])
        fitter.get_post(<double*> &post[0])
        fitter.get_log_probability(<double*> &log_probability[0])
        self.stats['trans'] = transition_counts
        self.stats['obs'] = obs
        self.stats['obs**2'] = obs2
        self.stats['post'] = post
        self.stats['log_probability'] = log_probability

    cdef _record_stats_double(self, GaussianHMMFitter[double]* fitter):
        """Copy various statistics from the C++ class to this one."""
        cdef np.ndarray[double, ndim=2] transition_counts
        cdef np.ndarray[double, ndim=2] obs
        cdef np.ndarray[double, ndim=2] obs2
        cdef np.ndarray[double, ndim=1] post
        cdef np.ndarray[double, ndim=1] log_probability
        transition_counts = np.empty((self.n_states, self.n_states))
        obs = np.empty((self.n_states, self.n_features))
        obs2 = np.empty((self.n_states, self.n_features))
        post = np.empty(self.n_states)
        log_probability = np.empty(fitter.get_fit_iterations())
        fitter.get_transition_counts(<double*> &transition_counts[0,0])
        fitter.get_obs(<double*> &obs[0,0])
        fitter.get_obs2(<double*> &obs2[0,0])
        fitter.get_post(<double*> &post[0])
        fitter.get_log_probability(<double*> &log_probability[0])
        self.stats['trans'] = transition_counts
        self.stats['obs'] = obs
        self.stats['obs**2'] = obs2
        self.stats['post'] = post
        self.stats['log_probability'] = log_probability

    def __reduce__(self):
        """Pickle support"""
        args = (self.n_states, self.n_init, self.n_iter, self.n_lqa_iter, self.fusion_prior, self.thresh,
                self.reversible_type, self.vars_prior, self.vars_weight, self.random_state,
                self.timing, self.n_hotstart, self.init_algo)
        state = (self._means_, self._vars_, self._transmat_, self._populations_, self._fit_logprob_, self._fit_time_, self.n_features)
        return (self.__class__, args, state)

    def __setstate__(self, state):
        """Pickle support"""
        self._means_ = state[0]
        self._vars_ = state[1]
        self._transmat_ = state[2]
        self._populations_ = state[3]
        self._fit_logprob_ = state[4]
        self._fit_time_ = state[5]
        self.n_features = state[6]

cdef public void _do_mstep_float(GaussianHMM hmm, GaussianHMMFitter[float]* fitter):
    """This function exists to let the C++ code call back into Cython."""
    cdef np.ndarray[double, ndim=2] transmat
    cdef np.ndarray[double, ndim=2] means
    cdef np.ndarray[double, ndim=2] vars
    hmm._record_stats_float(fitter)
    hmm._do_mstep()
    transmat = hmm._transmat_
    means = hmm._means_.astype(np.float64)
    vars = hmm._vars_.astype(np.float64)
    fitter.set_transmat(<double*> &transmat[0,0])
    fitter.set_means_and_variances(<double*> &means[0,0], <double*> &vars[0,0])

cdef public void _do_mstep_double(GaussianHMM hmm, GaussianHMMFitter[double]* fitter):
    """This function exists to let the C++ code call back into Cython."""
    cdef np.ndarray[double, ndim=2] transmat
    cdef np.ndarray[double, ndim=2] means
    cdef np.ndarray[double, ndim=2] vars
    hmm._record_stats_double(fitter)
    hmm._do_mstep()
    transmat = hmm._transmat_
    means = hmm._means_.astype(np.float64)
    vars = hmm._vars_.astype(np.float64)
    fitter.set_transmat(<double*> &transmat[0,0])
    fitter.set_means_and_variances(<double*> &means[0,0], <double*> &vars[0,0])
