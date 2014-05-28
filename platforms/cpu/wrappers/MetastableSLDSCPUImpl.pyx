import numpy as np
from sklearn.hmm import GaussianHMM


from libcpp cimport bool
cimport numpy as np
from libc.stdlib cimport malloc, free

cdef extern from "mslds_estep.hpp" namespace "Mixtape":
    void do_estep_single "Mixtape::do_mslds_estep<float>"(
        const float* log_transmat, const float* log_transmat_T,
        const float* log_startprob, const float* As, const float* bs,
        const float* Qs, const float* means,
        const float* covariances, const float** sequences,
        const int n_sequences, const int* sequence_lengths,
        const int n_features, const int n_states, const bool hmm_hotstart,
        float* transcounts, float* obs, float* obs_but_first,
        float* obs_but_last, float* obs_obs_t, float* obs_obs_T_offset,
        float* obs_obs_T_but_first, float* obs_obs_T_but_last,
        float* post, float* post_but_first, float* post_but_last,
        float* logprob) nogil


    void do_estep_mixed "Mixtape::do_mslds_estep<double>"(
        const float* log_transmat, const float* log_transmat_T,
        const float* log_startprob, const float* As, const float* bs,
        const float* Qs, const float* means,
        const float* covariances, const float** sequences,
        const int n_sequences, const int* sequence_lengths,
        const int n_features, const int n_states, const bool hmm_hotstart,
        float* transcounts, float* obs, float* obs_but_first,
        float* obs_but_last, float* obs_obs_t, float* obs_obs_T_offset,
        float* obs_obs_T_but_first, float* obs_obs_T_but_last,
        float* post, float* post_but_first, float* post_but_last,
        float* logprob) nogil


cdef class MetastableSLDSCPUImpl:
    cdef list sequences
    cdef int n_sequences
    cdef np.ndarray seq_lengths
    cdef int n_states, n_features
    cdef str precision
    cdef np.ndarray means, covars, Qs, As, bs
    cdef log_transmat, log_transmat_T, log_startprob, 

    def __cinit__(self, n_states, n_features, precision='single'):
        self.n_states = n_states
        self.n_features = n_features
        self.precision = str(precision)
        if self.precision not in ['single', 'mixed']:
            raise ValueError('This platform only supports ',
            'single or mixed precision')            

    property _sequences:
        def __set__(self, value):
            self.sequences = value
            self.n_sequences = len(value)
            if self.n_sequences <= 0:
                raise ValueError('More than 0 sequences must be provided')

            cdef np.ndarray[ndim=1, dtype=np.int32_t] seq_lengths = \
                    np.zeros(self.n_sequences, dtype=np.int32)
            for i in range(self.n_sequences):
                self.sequences[i] = np.asarray(self.sequences[i],
                        order='c', dtype=np.float32)
                seq_lengths[i] = len(self.sequences[i])
                # print np.shape(self.sequences[i])
                if self.n_features != self.sequences[i].shape[1]:
                    raise ValueError(('All sequences must be arrays '
                                      'of shape N by %d') 
                                      % self.n_features)
            self.seq_lengths = seq_lengths

    property means_:
        def __set__(self, np.ndarray[ndim=2, 
                                dtype=np.float32_t, mode='c'] m):
            if ((m.shape[0] != self.n_states) 
                    or (m.shape[1] != self.n_features)):
                raise TypeError(('Means must have shape (%d, %d), '
                        'You supplied (%d, %d)') 
                    % (self.n_states, self.n_features, 
                        m.shape[0], m.shape[1]))
            self.means = m
        
        def __get__(self):
            return self.means

    property bs_:
        def __set__(self, np.ndarray[ndim=2, 
                        dtype=np.float32_t, mode='c'] b):
            if ((b.shape[0] != self.n_states) 
                    or (b.shape[1] != self.n_features)):
                raise TypeError(('Means must have shape (%d, %d), '
                        'You supplied (%d, %d)') %
                                (self.n_states, self.n_features, b
                                    .shape[0], b.shape[1]))
            self.bs = b
        
        def __get__(self):
            return self.bs

    property covars_:
        def __set__(self, np.ndarray[ndim=3, 
                        dtype=np.float32_t, mode='c'] v):
            if ((v.shape[0] != self.n_states) 
                    or (v.shape[1] != self.n_features) 
                    or (v.shape[2] != self.n_features)):
                raise TypeError(('Variances must have shape (%d, %d, %d),'
                                ' You supplied (%d, %d, %d)') 
                    % (self.n_states, self.n_features, self.n_features, 
                        v.shape[0], v.shape[1], v.shape[1]))
            self.covars = v
        
        def __get__(self):
            return self.covars

    property Qs_:
        def __set__(self, np.ndarray[ndim=3, 
                        dtype=np.float32_t, mode='c'] q):
            if ((q.shape[0] != self.n_states) 
                    or (q.shape[1] != self.n_features) 
                    or (q.shape[2] != self.n_features)):
                raise TypeError(('Local variances must have shape '
                        '(%d, %d, %d), You supplied (%d, %d, %d)') 
                    % (self.n_states, self.n_features, self.n_features, 
                        q.shape[0], q.shape[1], q.shape[1]))
            self.Qs = q
        
        def __get__(self):
            return self.Qs

    property As_:
        def __set__(self, np.ndarray[ndim=3, 
                        dtype=np.float32_t, mode='c'] a):
            if ((a.shape[0] != self.n_states) 
                    or (a.shape[1] != self.n_features) 
                    or (a.shape[2] != self.n_features)):
                raise TypeError(('Local variances must have shape '
                        '(%d, %d, %d), You supplied (%d, %d, %d)') 
                    % (self.n_states, self.n_features, self.n_features, 
                        a.shape[0], a.shape[1], a.shape[1]))
            self.As = a
        
        def __get__(self):
            return self.As
    
    property transmat_:
        def __set__(self, np.ndarray[ndim=2, 
                        dtype=np.float32_t, mode='c'] t):
            if ((t.shape[0] != self.n_states) 
                    or (t.shape[1] != self.n_states)):
                raise TypeError(('transmat must have shape (%d, %d), '
                                 'You supplied (%d, %d)') 
                             % (self.n_states, self.n_states, 
                                 t.shape[0], t.shape[1]))
            self.log_transmat = np.log(t)
            self.log_transmat_T = np.asarray(self.log_transmat.T,
                                            order='C')
        
        def __get__(self):
            return np.exp(self.log_transmat)
    
    property startprob_:
        def __get__(self):
            return np.exp(self.log_startprob)
    
        def __set__(self, np.ndarray[ndim=1, 
                        dtype=np.float32_t, mode='c'] s):
            if (s.shape[0] != self.n_states):
                raise TypeError(('startprob must have shape (%d,), '
                                 'You supplied (%d,)') %
                                (self.n_states, s.shape[0]))
            self.log_startprob = np.log(s)

    def do_mslds_estep(self):
        return self.do_estep(hmm_hotstart=False)

    def do_hmm_estep(self):
        return self.do_estep(hmm_hotstart=True)
    
    def do_estep(self, hmm_hotstart=False):
        cdef np.ndarray[ndim=2, mode='c', 
                dtype=np.float32_t] log_transmat = self.log_transmat
        cdef np.ndarray[ndim=2, mode='c', 
                dtype=np.float32_t] log_transmat_T = self.log_transmat_T
        cdef np.ndarray[ndim=1, mode='c', 
                dtype=np.float32_t] log_startprob = self.log_startprob
        cdef np.ndarray[ndim=3, mode='c', 
                dtype=np.float32_t] As = self.As
        cdef np.ndarray[ndim=2, mode='c', 
                dtype=np.float32_t] bs = self.bs
        cdef np.ndarray[ndim=3, mode='c', 
                dtype=np.float32_t] Qs = self.Qs
        cdef np.ndarray[ndim=2, mode='c', 
                dtype=np.float32_t] means = self.means
        cdef np.ndarray[ndim=3, mode='c', 
                dtype=np.float32_t] covariances = self.covars
        cdef np.ndarray[ndim=1, mode='c', 
                dtype=np.int32_t] seq_lengths = self.seq_lengths

        # All of the sufficient statistics
        cdef np.ndarray[ndim=2, mode='c', dtype=np.float32_t] \
                transcounts = np.zeros((self.n_states, self.n_states), 
                        dtype=np.float32)
        cdef np.ndarray[ndim=2, mode='c', dtype=np.float32_t] \
                obs = np.zeros((self.n_states, self.n_features), 
                        dtype=np.float32)
        cdef np.ndarray[ndim=2, mode='c', dtype=np.float32_t] \
                obs_but_first = np.zeros((self.n_states, self.n_features),
                        dtype=np.float32)
        cdef np.ndarray[ndim=2, mode='c', dtype=np.float32_t] \
                obs_but_last = np.zeros((self.n_states, self.n_features), 
                        dtype=np.float32)

        cdef np.ndarray[ndim=3, mode='c', dtype=np.float32_t] \
                obs_obs_T = np.zeros((self.n_states, self.n_features,
                    self.n_features), dtype=np.float32)
        cdef np.ndarray[ndim=3, mode='c', dtype=np.float32_t] \
                obs_obs_T_offset = np.zeros((self.n_states, 
                    self.n_features, self.n_features), dtype=np.float32)
        cdef np.ndarray[ndim=3, mode='c', dtype=np.float32_t] \
                obs_obs_T_but_first = np.zeros((self.n_states, 
                    self.n_features, self.n_features), dtype=np.float32)
        cdef np.ndarray[ndim=3, mode='c', dtype=np.float32_t] \
                obs_obs_T_but_last = np.zeros((self.n_states, 
                    self.n_features, self.n_features), dtype=np.float32)

        cdef np.ndarray[ndim=1, mode='c', dtype=np.float32_t] \
                post = np.zeros(self.n_states, dtype=np.float32)
        cdef np.ndarray[ndim=1, mode='c', dtype=np.float32_t] \
                post_but_first = np.zeros(self.n_states, dtype=np.float32)
        cdef np.ndarray[ndim=1, mode='c', dtype=np.float32_t] \
                post_but_last = np.zeros(self.n_states, dtype=np.float32)
        cdef float logprob

        seq_pointers = <float**> malloc(self.n_sequences * sizeof(float*))
        cdef np.ndarray[ndim=2, mode='c', dtype=np.float32_t] sequence
        for i in range(self.n_sequences):
            sequence = self.sequences[i]
            seq_pointers[i] = &sequence[0,0]


        if self.precision == 'single':
            do_estep_single(
                <float*> &log_transmat[0,0], 
                <float*> &log_transmat_T[0,0], <float*> &log_startprob[0],
                <float*> &As[0,0,0], <float*> &bs[0,0], 
                <float*> &Qs[0,0,0], <float*> &means[0,0],
                <float*> &covariances[0,0,0],
                <const float**> seq_pointers,
                self.n_sequences, <int*> &seq_lengths[0], self.n_features,
                self.n_states, hmm_hotstart,
                <float*> &transcounts[0,0], <float*>
                &obs[0,0], <float*> &obs_but_first[0,0],
                <float*> &obs_but_last[0,0], <float*> &obs_obs_T[0,0,0],
                <float*> &obs_obs_T_offset[0,0,0],
                <float*> &obs_obs_T_but_first[0,0,0],
                <float*> &obs_obs_T_but_last[0,0,0], <float*> &post[0],
                <float*> &post_but_first[0], <float*> &post_but_last[0],
                &logprob)
        elif self.precision == 'mixed':
            do_estep_mixed(
                <float*> &log_transmat[0,0], 
                <float*> &log_transmat_T[0,0], <float*> &log_startprob[0],
                <float*> &As[0,0,0], <float*> &bs[0,0], 
                <float*> &Qs[0,0,0], <float*> &means[0,0],
                <float*> &covariances[0,0,0], 
                <const float**> seq_pointers, 
                self.n_sequences, <int*> &seq_lengths[0], 
                self.n_features, self.n_states, hmm_hotstart,
                <float*> &transcounts[0,0], <float*> &obs[0,0],
                <float*> &obs_but_first[0,0], <float*> &obs_but_last[0,0],
                <float*> &obs_obs_T[0,0,0], 
                <float*> &obs_obs_T_offset[0,0,0],
                <float*> &obs_obs_T_but_first[0,0,0],
                <float*> &obs_obs_T_but_last[0,0,0], <float*> &post[0],
                <float*> &post_but_first[0], <float*> &post_but_last[0],
                &logprob)
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

###############################################################################
# Tests. These are exposed to nose by being called from one of the python
# test files
###############################################################################

cdef extern from "gaussian_likelihood.h":
    void gaussian_loglikelihood_full(const float* sequence,
                                     const float* means,
                                     const float* covariances,
                                     const int n_observations,
                                     const int n_states,
                                     const int n_features,
                                     float* loglikelihoods)

def test_gaussian_loglikelihood_full():
    # check gaussian_loglikelihood_full vs. a reference python implementation

    from sklearn.mixture.gmm import _log_multivariate_normal_density_full

    cdef int length = 5
    cdef int n_states = 2
    cdef int n_features = 3
    
    cdef np.ndarray[ndim=2, mode='c', dtype=np.float32_t] sequence \
            = np.random.randn(length, n_features).astype(np.float32)
    cdef np.ndarray[ndim=2, mode='c', dtype=np.float32_t] means \
            = np.random.randn(n_states, n_features).astype(np.float32)
    cdef np.ndarray[ndim=3, mode='c', dtype=np.float32_t] covariances \
            = (np.random.rand(n_states, n_features, n_features)
                    .astype(np.float32))
    for i in range(n_states):
        covariances[i] += (covariances[i].T 
                + 10*np.eye(n_features, n_features))
    cdef np.ndarray[ndim=2, mode='c', dtype=np.float32_t] loglikelihoods\
            = np.zeros((length, n_states), dtype=np.float32)
    
    val = _log_multivariate_normal_density_full(sequence, 
                    means, covariances)
    gaussian_loglikelihood_full(&sequence[0, 0], &means[0, 0],
            &covariances[0, 0,0], length, n_states, 
            n_features, &loglikelihoods[0, 0]);
    np.testing.assert_array_almost_equal(val, loglikelihoods)
