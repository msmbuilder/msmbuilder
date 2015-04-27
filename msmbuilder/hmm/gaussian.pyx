import numpy as np
cimport numpy as np
from libcpp.vector cimport vector
from libcpp.string cimport string

import warnings
from sklearn import cluster, mixture
from mdtraj.utils import ensure_type
from ..utils import check_iter_of_sequences
from ..msm._markovstatemodel import _transmat_mle_prinz

cdef extern from "Trajectory.h" namespace "Mixtape":
    cdef cppclass Trajectory:
        Trajectory(char*, int, int, int, int) except +

cdef extern from "GaussianHMMFitter.h" namespace "Mixtape":
    cdef cppclass GaussianHMMFitter[T]:
        GaussianHMMFitter(GaussianHMM, int, int, int, double*) except +
        void set_transmat(double*)
        void set_means_and_variances(double*, double*)
        void fit(const vector[Trajectory], double)
        void get_transition_counts(double*)
        void get_obs(double*)
        void get_obs2(double*)
        void get_post(double*)

cdef public class GaussianHMM[object GaussianHMMObject, type GaussianHMMType]:
    cdef int n_states, n_features, n_iter
    cdef float thresh
    cdef startprob
    cdef stats
    cdef reversible_type, n_lqa_iter, fusion_prior, transmat_prior, vars_prior, vars_weight, init_algo
    cdef _means_, _vars_, _transmat_, _populations_

    def __init__(self, n_states, n_features, n_iter=10, thresh=1e-2):
        self.n_states = int(n_states)
        self.n_features = int(n_features)
        self.n_iter = int(n_iter)
        self.thresh = float(thresh)
        self.startprob = np.tile(1.0/n_states, n_states)
        self._transmat_ = np.empty((n_states, n_states))
        self._transmat_.fill(1.0/n_states)
        self.stats = {}
        
        self.reversible_type = 'mle'
        self.n_lqa_iter = 10
        self.fusion_prior = 1e-2
        self.transmat_prior = 1.0
        self.vars_prior = 1e-3
        self.vars_weight = 1
        self.init_algo = "kmeans"

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
    
    def fit(self, sequences):
        if len(sequences) == 0:
            raise ValueError('sequences is empty')
        check_iter_of_sequences(sequences)
        dtype = sequences[0].dtype
        if any(s.dtype != dtype for s in sequences):
            raise ValueError('All sequences must have the same data type')
        if any(s.shape[1] != self.n_features for s in sequences):
            raise ValueError('All sequences must have %d features' % self.n_features)
        self.stats = {}
        self._init(sequences)
        if dtype == np.float32:
            self._fit_float(sequences)
        elif dtype == np.float64:
            self._fit_double(sequences)
        else:
            raise ValueError('Unsupported data type: '+str(dtype))

    def _init(self, sequences):
        """Find initial means(hot start)"""
        sequences = [ensure_type(s, dtype=np.float32, ndim=2, name='s', warn_on_cast=False)
                     for s in sequences]
        #self._impl._sequences = sequences

        small_dataset = np.vstack(sequences)

        if self.init_algo == "GMM":
            mix = mixture.GMM(self.n_states, n_init=1, random_state=None)
            mix.fit(small_dataset)
            self._means_ = mix.means_
            self._vars_ = mix.covars_
        else:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                self._means_ = cluster.KMeans(
                    n_clusters=self.n_states, n_init=1, init='random',
                    n_jobs=1, random_state=None).fit(
                    small_dataset).cluster_centers_
            self._vars_ = np.vstack([np.var(small_dataset, axis=0)] * self.n_states)
        self._populations_ = np.ones(self.n_states) / self.n_states
        
    def _fit_float(self, sequences):
        cdef vector[Trajectory] trajectoryVec
        cdef np.ndarray[float, ndim=2] array
        cdef np.ndarray[double, ndim=1] startprob
        cdef np.ndarray[double, ndim=2] transmat
        cdef np.ndarray[double, ndim=2] means
        cdef np.ndarray[double, ndim=2] vars
        startprob = self.startprob
        transmat = self._transmat_
        means = self._means_.astype(np.float64)
        vars = self._vars_.astype(np.float64)
        for s in sequences:
            array = s
            trajectoryVec.push_back(Trajectory(<char*> &array[0,0], array.shape[0], array.shape[1], array.strides[0], array.strides[1]))
        cdef GaussianHMMFitter[float] *fitter = new GaussianHMMFitter[float](self, self.n_states, self.n_features, self.n_iter, <double*> &startprob[0])
        fitter.set_transmat(<double*> &transmat[0,0])
        fitter.set_means_and_variances(<double*> &means[0,0], <double*> &vars[0,0])
        try:
            fitter.fit(trajectoryVec, self.thresh)
        finally:
            del fitter
        
    def _fit_double(self, sequences):
        cdef vector[Trajectory] trajectoryVec
        cdef np.ndarray[double, ndim=2] array
        cdef np.ndarray[double, ndim=1] startprob
        cdef np.ndarray[double, ndim=2] transmat
        cdef np.ndarray[double, ndim=2] means
        cdef np.ndarray[double, ndim=2] vars
        startprob = self.startprob
        transmat = self._transmat_
        means = self._means_.astype(np.float64)
        vars = self._vars_.astype(np.float64)
        for s in sequences:
            array = s
            trajectoryVec.push_back(Trajectory(<char*> &array[0,0], array.shape[0], array.shape[1], array.strides[0], array.strides[1]))
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
        if self.reversible_type == 'mle':
            counts = np.maximum(
                np.nan_to_num(stats['trans']) + self.transmat_prior - 1.0,
                    1e-20).astype(np.float64)
            self._transmat_, self._populations_ = _transmat_mle_prinz(counts)
        elif self.reversible_type == 'transpose':
            revcounts = np.maximum(
                self.transmat_prior - 1.0 + stats['trans'] + stats['trans'].T, 1e-20)
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
    
    cdef _record_stats_float(self, GaussianHMMFitter[float]* fitter):
        cdef np.ndarray[double, ndim=2] transition_counts
        cdef np.ndarray[double, ndim=2] obs
        cdef np.ndarray[double, ndim=2] obs2
        cdef np.ndarray[double, ndim=1] post
        transition_counts = np.empty((self.n_states, self.n_states))
        obs = np.empty((self.n_states, self.n_features))
        obs2 = np.empty((self.n_states, self.n_features))
        post = np.empty(self.n_states)
        fitter.get_transition_counts(<double*> &transition_counts[0,0])
        fitter.get_obs(<double*> &obs[0,0])
        fitter.get_obs2(<double*> &obs2[0,0])
        fitter.get_post(<double*> &post[0])
        self.stats['trans'] = transition_counts
        self.stats['obs'] = obs
        self.stats['obs**2'] = obs2
        self.stats['post'] = post
    
    cdef _record_stats_double(self, GaussianHMMFitter[double]* fitter):
        cdef np.ndarray[double, ndim=2] transition_counts
        cdef np.ndarray[double, ndim=2] obs
        cdef np.ndarray[double, ndim=2] obs2
        cdef np.ndarray[double, ndim=1] post
        transition_counts = np.empty((self.n_states, self.n_states))
        obs = np.empty((self.n_states, self.n_features))
        obs2 = np.empty((self.n_states, self.n_features))
        post = np.empty(self.n_states)
        fitter.get_transition_counts(<double*> &transition_counts[0,0])
        fitter.get_obs(<double*> &obs[0,0])
        fitter.get_obs2(<double*> &obs2[0,0])
        fitter.get_post(<double*> &post[0])
        self.stats['trans'] = transition_counts
        self.stats['obs'] = obs
        self.stats['obs**2'] = obs2
        self.stats['post'] = post

cdef public void _do_mstep_float(GaussianHMM hmm, GaussianHMMFitter[float]* fitter):
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
