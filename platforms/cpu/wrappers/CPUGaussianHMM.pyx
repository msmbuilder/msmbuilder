import numpy as np
cimport numpy as np
from cython.parallel cimport prange
from libc.stdlib cimport malloc, free


cdef extern from "do_estep.c":
    void do_estep(
        const float* log_transmat,
        const float* log_transmat_T,
        const float* log_startprob,
        const float* means,
        const float* variances,
        const float** sequences,
        const int n_sequences,
        const np.int32_t* sequence_lengths,
        const int n_features,
        const int n_states,
        float* transcounts,
        float* obs,
        float* obs2,
        float* post) nogil


cdef class CPUGaussianHMM:
    cdef list sequences
    cdef int n_sequences
    cdef np.ndarray seq_lengths
    cdef int n_states, n_features
    cdef float** seq_pointers
    cdef np.ndarray means, variances, log_transmat, log_transmat_T, log_startprob

    def __cinit__(self, n_states, n_features):
        self.n_states = n_states
        self.n_features = n_features

    property _sequences:
        def __set__(self, value):
            self.sequences = value
            self.n_sequences = len(value)
            if self.n_sequences <= 0:
                raise ValueError('More than 0 sequences must be provided')
            self.seq_pointers = <float**>malloc(self.n_sequences * sizeof(float*))
            
            cdef np.ndarray[ndim=1, dtype=np.int32_t] seq_lengths = np.zeros(self.n_sequences, dtype=np.int32)
            cdef np.ndarray[ndim=2, dtype=np.float32_t] S
            for i in range(self.n_sequences):
                self.sequences[i] = np.asarray(self.sequences[i], order='c',
                                               dtype=np.float32)
                S = self.sequences[i]
                self.seq_pointers[i] = &S[0,0]
                seq_lengths[i] = len(S)
                if self.n_features != S.shape[1]:
                    raise ValueError('All sequences must be arrays of shape N by %d' %
                                     self.n_features)
            self.seq_lengths = seq_lengths
        
        def __del__(self):
            free(self.seq_pointers)
            self.sequences = None
            self.n_sequences = 0

    property means_:
        def __set__(self, value):
            cdef np.ndarray[ndim=2, mode='c', dtype=np.float32_t] m = np.asarray(value, order='c', dtype=np.float32)
            if (m.shape[0] != self.n_states) or (m.shape[1] != self.n_features):
                raise TypeError('Means must have shape (%d, %d), You supplied (%d, %d)' %
                                (self.n_states, self.n_features, m.shape[0], m.shape[1]))
            self.means = m
        
        def __get__(self):
            return self.means

    property variances_:
        def __set__(self, value):
            cdef np.ndarray[ndim=2, mode='c', dtype=np.float32_t] v = np.asarray(value, order='c', dtype=np.float32)
            if (v.shape[0] != self.n_states) or (v.shape[1] != self.n_features):
                raise TypeError('Variances must have shape (%d, %d), You supplied (%d, %d)' %
                                (self.n_states, self.n_features, v.shape[0], v.shape[1]))
            self.variances = v
        
        def __get__(self):
            return self.variances
    
    property transmat_:
        def __set__(self, value):
            cdef np.ndarray[ndim=2, mode='c', dtype=np.float32_t] t = np.asarray(value, order='c', dtype=np.float32)
            if (t.shape[0] != self.n_states) or (t.shape[1] != self.n_states):
                raise TypeError('transmat must have shape (%d, %d), You supplied (%d, %d)' %
                                (self.n_states, self.n_states, t.shape[0], t.shape[1]))
            self.log_transmat = t
            self.log_transmat_T = np.asarray(t.T, order='C')
        
        def __get__(self):
            return np.exp(self.log_transmat)
    
    property startprob_:
        def __get__(self):
            return np.exp(self.startprob)
    
        def __set__(self, value):
            cdef np.ndarray[ndim=1, mode='c', dtype=np.float32_t] s = np.asarray(value, order='c', dtype=np.float32)
            if (s.shape[0] != self.n_states):
                raise TypeError('startprob must have shape (%d,), You supplied (%d,)' %
                                (self.n_states, s.shape[0]))
            self.log_startprob = np.log(s)

    def fit(self, sequences):
        self._sequences = sequences
        for i in range(2):
            self._do_estep()

    def _do_estep(self):
        cdef np.ndarray[ndim=2, mode='c', dtype=np.float32_t] log_transmat = self.log_transmat
        cdef np.ndarray[ndim=2, mode='c', dtype=np.float32_t] log_transmat_T = self.log_transmat_T
        cdef np.ndarray[ndim=1, mode='c', dtype=np.float32_t] log_startprob = self.log_startprob
        cdef np.ndarray[ndim=2, mode='c', dtype=np.float32_t] means = self.means
        cdef np.ndarray[ndim=2, mode='c', dtype=np.float32_t] variances = self.variances
        cdef np.ndarray[ndim=1, mode='c', dtype=np.int32_t] seq_lengths = self.seq_lengths

        cdef np.ndarray[ndim=2, mode='c', dtype=np.float32_t] transcounts = np.zeros((self.n_states, self.n_states), dtype=np.float32)
        cdef np.ndarray[ndim=2, mode='c', dtype=np.float32_t] obs = np.zeros((self.n_states, self.n_features), dtype=np.float32)
        cdef np.ndarray[ndim=2, mode='c', dtype=np.float32_t] obs2 = np.zeros((self.n_states, self.n_features), dtype=np.float32)
        cdef np.ndarray[ndim=1, mode='c', dtype=np.float32_t] post = np.zeros(self.n_states, dtype=np.float32)

        cdef float** seq_pointers = self.seq_pointers
        
        print "Calling do_estep"
        do_estep(&log_transmat[0,0], &log_transmat_T[0,0], &log_startprob[0], &means[0,0], &variances[0,0],
                 <const float**> seq_pointers, self.n_sequences, &seq_lengths[0], self.n_features, self.n_states,
                 &transcounts[0,0], &obs[0,0], &obs2[0,0], &post[0])
        print "Returned from do_estep"

        


    def __dealloc__(self):
        del self._sequences
