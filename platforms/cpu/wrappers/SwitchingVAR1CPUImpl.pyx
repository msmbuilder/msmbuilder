import numpy as np
from sklearn.hmm import GaussianHMM


cimport numpy as np
from libc.stdlib cimport malloc, free

cdef extern from "mslds_estep.hpp" namespace "Mixtape":
    void do_estep_single "Mixtape::do_mslds_estep<float>"(
        const float* log_transmat, const float* log_transmat_T,
        const float* log_startprob, const float* means,
        const float* covariances, const float** sequences,
        const int n_sequences, const np.int32_t* sequence_lengths,
        const int n_features, const int n_states,
        float* transcounts, float* obs, float* obs_but_first,
        float* obs_but_last, float* obs_obs_t, float* obs_obs_T_offset,
        float* obs_obs_T_but_first, float* obs_obs_T_but_last,
        float* post, float* post_but_first, float* post_but_last,
        float* logprob) nogil
    
    void do_estep_mixed "Mixtape::do_mslds_estep<double>"(
        const float* log_transmat, const float* log_transmat_T,
        const float* log_startprob, const float* means,
        const float* covariances, const float** sequences,
        const int n_sequences, const np.int32_t* sequence_lengths,
        const int n_features, const int n_states,
        float* transcounts, float* obs, float* obs_but_first,
        float* obs_but_last, float* obs_obs_t, float* obs_obs_T_offset,
        float* obs_obs_T_but_first, float* obs_obs_T_but_last,
        float* post, float* post_but_first, float* post_but_last,
        float* logprob) nogil

cdef extern from "gaussian_likelihood.h":
    void gaussian_loglikelihood_full(const float* sequence,
                                     const float* means,
                                     const float* covariances,
                                     const int n_observations,
                                     const int n_states,
                                     const int n_features,
                                     float* loglikelihoods)


cdef class SwitchingVAR1CPUImpl:
    cdef list sequences
    cdef int n_sequences
    cdef np.ndarray seq_lengths
    cdef int n_states, n_features
    cdef str precision
    cdef np.ndarray means, covars, log_transmat, log_transmat_T, log_startprob

    def __cinit__(self, n_states, n_features, precision='single'):
        self.n_states = n_states
        self.n_features = n_features
        self.precision = str(precision)
        if self.precision not in ['single', 'mixed']:
            raise ValueError('This platform only supports single or mixed precision')            

    property _sequences:
        def __set__(self, value):
            self.sequences = value
            self.n_sequences = len(value)
            if self.n_sequences <= 0:
                raise ValueError('More than 0 sequences must be provided')

            cdef np.ndarray[ndim=1, dtype=np.int32_t] seq_lengths = np.zeros(self.n_sequences, dtype=np.int32)
            cdef np.ndarray[ndim=2, dtype=np.float32_t] S
            for i in range(self.n_sequences):
                self.sequences[i] = np.asarray(self.sequences[i], order='c', dtype=np.float32)
                S = self.sequences[i]
                seq_lengths[i] = len(S)
                if self.n_features != S.shape[1]:
                    raise ValueError('All sequences must be arrays of shape N by %d' %
                                     self.n_features)
            self.seq_lengths = seq_lengths

    property means_:
        def __set__(self, np.ndarray[ndim=2, dtype=np.float32_t, mode='c'] m):
            if (m.shape[0] != self.n_states) or (m.shape[1] != self.n_features):
                raise TypeError('Means must have shape (%d, %d), You supplied (%d, %d)' %
                                (self.n_states, self.n_features, m.shape[0], m.shape[1]))
            self.means = m
        
        def __get__(self):
            return self.means

    property covars_:
        def __set__(self, np.ndarray[ndim=3, dtype=np.float32_t, mode='c'] v):
            if (v.shape[0] != self.n_states) or (v.shape[1] != self.n_features) or (v.shape[2] != self.n_features):
                raise TypeError('Variances must have shape (%d, %d, %d), You supplied (%d, %d, %d)' %
                                (self.n_states, self.n_features, self.n_features, v.shape[0], v.shape[1], v.shape[1]))
            self.covars = v
        
        def __get__(self):
            return self.covars
    
    property transmat_:
        def __set__(self, np.ndarray[ndim=2, dtype=np.float32_t, mode='c'] t):
            if (t.shape[0] != self.n_states) or (t.shape[1] != self.n_states):
                raise TypeError('transmat must have shape (%d, %d), You supplied (%d, %d)' %
                                (self.n_states, self.n_states, t.shape[0], t.shape[1]))
            self.log_transmat = np.log(t)
            self.log_transmat_T = np.asarray(self.log_transmat.T, order='C')
        
        def __get__(self):
            return np.exp(self.log_transmat)
    
    property startprob_:
        def __get__(self):
            return np.exp(self.log_startprob)
    
        def __set__(self, np.ndarray[ndim=1, dtype=np.float32_t, mode='c'] s):
            if (s.shape[0] != self.n_states):
                raise TypeError('startprob must have shape (%d,), You supplied (%d,)' %
                                (self.n_states, s.shape[0]))
            self.log_startprob = np.log(s)

    
    def do_estep(self):
        #starttime = time.time()
        cdef np.ndarray[ndim=2, mode='c', dtype=np.float32_t] log_transmat = self.log_transmat
        cdef np.ndarray[ndim=2, mode='c', dtype=np.float32_t] log_transmat_T = self.log_transmat_T
        cdef np.ndarray[ndim=1, mode='c', dtype=np.float32_t] log_startprob = self.log_startprob
        cdef np.ndarray[ndim=2, mode='c', dtype=np.float32_t] means = self.means
        cdef np.ndarray[ndim=3, mode='c', dtype=np.float32_t] covars = self.covars
        cdef np.ndarray[ndim=1, mode='c', dtype=np.int32_t] seq_lengths = self.seq_lengths

        # All of the sufficient statistics
        cdef np.ndarray[ndim=2, mode='c', dtype=np.float32_t] transcounts = np.zeros((self.n_states, self.n_states), dtype=np.float32)

        cdef np.ndarray[ndim=2, mode='c', dtype=np.float32_t] obs = np.zeros((self.n_states, self.n_features), dtype=np.float32)
        cdef np.ndarray[ndim=2, mode='c', dtype=np.float32_t] obs_but_first = np.zeros((self.n_states, self.n_features), dtype=np.float32)
        cdef np.ndarray[ndim=2, mode='c', dtype=np.float32_t] obs_but_last = np.zeros((self.n_states, self.n_features), dtype=np.float32)

        cdef np.ndarray[ndim=3, mode='c', dtype=np.float32_t] obs_obs_T = np.zeros((self.n_states, self.n_features, self.n_features), dtype=np.float32)
        cdef np.ndarray[ndim=3, mode='c', dtype=np.float32_t] obs_obs_T_offset = np.zeros((self.n_states, self.n_features, self.n_features), dtype=np.float32)
        cdef np.ndarray[ndim=3, mode='c', dtype=np.float32_t] obs_obs_T_but_first = np.zeros((self.n_states, self.n_features, self.n_features), dtype=np.float32)
        cdef np.ndarray[ndim=3, mode='c', dtype=np.float32_t] obs_obs_T_but_last = np.zeros((self.n_states, self.n_features, self.n_features), dtype=np.float32)

        cdef np.ndarray[ndim=1, mode='c', dtype=np.float32_t] post = np.zeros(self.n_states, dtype=np.float32)
        cdef np.ndarray[ndim=1, mode='c', dtype=np.float32_t] post_but_first = np.zeros(self.n_states, dtype=np.float32)
        cdef np.ndarray[ndim=1, mode='c', dtype=np.float32_t] post_but_last = np.zeros(self.n_states, dtype=np.float32)
        cdef float logprob

        seq_pointers = <float**>malloc(self.n_sequences * sizeof(float*))
        cdef np.ndarray[ndim=2, mode='c', dtype=np.float32_t] sequence
        for i in range(self.n_sequences):
            sequence = self.sequences[i]
            seq_pointers[i] = &sequence[0,0]

        if self.precision == 'single':
            do_estep_single(
                &log_transmat[0,0], &log_transmat_T[0,0], &log_startprob[0],
                &means[0,0], &covars[0,0,0], <const float**> seq_pointers,
                self.n_sequences, &seq_lengths[0], self.n_features,
                self.n_states, &transcounts[0,0], &obs[0,0], &obs_but_first[0,0],
                &obs_but_last[0,0], &obs_obs_T[0,0,0], &obs_obs_T_offset[0,0,0],
                &obs_obs_T_but_first[0,0,0], &obs_obs_T_but_last[0,0,0], &post[0],
                &post_but_first[0], &post_but_last[0], &logprob)
        elif self.precision == 'mixed':
            do_estep_mixed(
                &log_transmat[0,0], &log_transmat_T[0,0], &log_startprob[0], &means[0,0],
                &covars[0,0,0], <const float**> seq_pointers, self.n_sequences, &seq_lengths[0],
                self.n_features, self.n_states, &transcounts[0,0], &obs[0,0], &obs_but_first[0,0],
                &obs_but_last[0,0], &obs_obs_T[0,0,0], &obs_obs_T_offset[0,0,0],
                &obs_obs_T_but_first[0,0,0], &obs_obs_T_but_last[0,0,0], &post[0],
                &post_but_first[0], &post_but_last[0], &logprob)
        else:
            raise RuntimeError('Invalid precision')

        free(seq_pointers)
        result = {
            'trans': transcounts,
            'obs': obs,
            'obs[1:]' : obs_but_first,
            'obs[:-1]': obs_but_last,
            'obs*obs.T': obs_obs_T,
            'obs*obs[t-1].T': obs_obs_T_offset,
            'obs[1:]*obs[1:].T': obs_obs_T_but_first,
            'obs[:-1]*obs[:-1].T': obs_obs_T_but_last,
            'post': post,
            'post[1:]': post_but_first,
            'post[:-1]': post_but_last,
        }
        return logprob, result


def test_1():
    from sklearn.mixture.gmm import _log_multivariate_normal_density_full
    cdef int length = 5
    cdef int n_states = 2
    cdef int n_features = 3
    
    cdef np.ndarray[ndim=2, mode='c', dtype=np.float32_t] sequence = np.random.randn(length, n_features).astype(np.float32)
    cdef np.ndarray[ndim=2, mode='c', dtype=np.float32_t] means = np.random.randn(n_states, n_features).astype(np.float32)
    cdef np.ndarray[ndim=3, mode='c', dtype=np.float32_t] covariances = np.random.rand(n_states, n_features, n_features).astype(np.float32)
    for i in range(n_states):
        covariances[i] += covariances[i].T + 10*np.eye(n_features, n_features)
    cdef np.ndarray[ndim=2, mode='c', dtype=np.float32_t] loglikelihoods = np.zeros((length, n_states), dtype=np.float32)
    
    val = _log_multivariate_normal_density_full(sequence, means, covariances)
    print 'sklearn'
    print val

    gaussian_loglikelihood_full(&sequence[0, 0], &means[0, 0], &covariances[0, 0,0],
       length, n_states, n_features, &loglikelihoods[0, 0]);
    # 
    print 'gaussian_loglikelihood_full'
    print loglikelihoods




def test_2():
    class ExitMe(Exception):
        def __init__(self, value):
            self.value = value

    class MyGaussianHMM(GaussianHMM):
        def _accumulate_sufficient_statistics(self, stats, obs, framelogprob,
                                              posteriors, fwdlattice, bwdlattice,
                                              params):
            super(MyGaussianHMM, self)._accumulate_sufficient_statistics(stats,
                obs, framelogprob, posteriors, fwdlattice, bwdlattice, params)
            raise ExitMe(stats)


    cdef int length = 3
    cdef int n_states = 2
    cdef int n_features = 2
    sequences = [np.random.randn(length, n_features).astype(np.float32)]
    
    
    gmm = MyGaussianHMM(n_components=n_states, covariance_type='full')
    try:
        gmm.fit(sequences)
    except ExitMe as e:
        stats = e.value
    
    
    model = SwitchingVAR1CPUImpl(n_states, n_features)
    model._sequences = sequences
    model.means_ = gmm.means_.astype(np.float32)
    model.covars_ = gmm.covars_.astype(np.float32)
    model.transmat_ = gmm.transmat_.astype(np.float32)
    model.startprob_ = gmm.startprob_.astype(np.float32)
    
    lpr, myresult = model.do_estep()

    from pprint import pprint
    print 'gmm stats'
    pprint(stats)
    print 'my stats'
    pprint(myresult)
    
    
# test_1()
test_2()