#################################################################
#    Copyright (c) 2013, Stanford University and the Authors    #
#    Author: Robert McGibbon <rmcgibbo@gmail.com>               #
#    Contributors:                                              #
#                                                               #
#################################################################

#import time
import numpy as np
cimport numpy as np
from libc.stdlib cimport malloc, free

cdef extern from "ghmm_estep.hpp" namespace "Mixtape":
    void do_estep_single "Mixtape::do_ghmm_estep<float>"(
        const float* log_transmat, const float* log_transmat_T,
        const float* log_startprob, const float* means,
        const float* variances, const float** sequences,
        const int n_sequences, const int* sequence_lengths,
        const int n_features, const int n_states,
        float* transcounts, float* obs, float* obs2,
        float* post, float* logprob) nogil
    void do_estep_mixed "Mixtape::do_ghmm_estep<double>"(
        const float* log_transmat, const float* log_transmat_T,
        const float* log_startprob, const float* means,
        const float* variances, const float** sequences,
        const int n_sequences, const int* sequence_lengths,
        const int n_features, const int n_states,
        float* transcounts, float* obs, float* obs2,
        float* post, float* logprob) nogil

cdef class GaussianHMMCPUImpl:
    cdef list sequences
    cdef int n_sequences
    cdef np.ndarray seq_lengths
    cdef int n_states, n_features
    cdef str precision
    cdef np.ndarray means, vars, log_transmat, log_transmat_T, log_startprob

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
            
            cdef np.ndarray[ndim=1, dtype=int] seq_lengths = np.zeros(self.n_sequences, dtype=np.int32)
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

    property vars_:
        def __set__(self, np.ndarray[ndim=2, dtype=np.float32_t, mode='c'] v):
            if (v.shape[0] != self.n_states) or (v.shape[1] != self.n_features):
                raise TypeError('Variances must have shape (%d, %d), You supplied (%d, %d)' %
                                (self.n_states, self.n_features, v.shape[0], v.shape[1]))
            self.vars = v
        
        def __get__(self):
            return self.vars
    
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
        cdef np.ndarray[ndim=2, mode='c', dtype=np.float32_t] vars = self.vars
        cdef np.ndarray[ndim=1, mode='c', dtype=int] seq_lengths = self.seq_lengths

        cdef np.ndarray[ndim=2, mode='c', dtype=np.float32_t] transcounts = np.zeros((self.n_states, self.n_states), dtype=np.float32)
        cdef np.ndarray[ndim=2, mode='c', dtype=np.float32_t] obs = np.zeros((self.n_states, self.n_features), dtype=np.float32)
        cdef np.ndarray[ndim=2, mode='c', dtype=np.float32_t] obs2 = np.zeros((self.n_states, self.n_features), dtype=np.float32)
        cdef np.ndarray[ndim=1, mode='c', dtype=np.float32_t] post = np.zeros(self.n_states, dtype=np.float32)
        cdef float logprob

        seq_pointers = <float**>malloc(self.n_sequences * sizeof(float*))
        cdef np.ndarray[ndim=2, mode='c', dtype=np.float32_t] sequence
        for i in range(self.n_sequences):
            sequence = self.sequences[i]
            seq_pointers[i] = &sequence[0,0]

        if self.precision == 'single':
            do_estep_single(
                <float*> &log_transmat[0,0], 
                <float*> &log_transmat_T[0,0], 
                <float*> &log_startprob[0], <float*> &means[0,0],
                <float*> &vars[0,0], <const float**> seq_pointers,
                self.n_sequences, <int*> &seq_lengths[0], self.n_features,
                self.n_states, <float*> &transcounts[0,0], 
                <float*> &obs[0,0], <float*> &obs2[0,0], 
                <float*> &post[0], &logprob)
        elif self.precision == 'mixed':
            do_estep_mixed(
                <float*> &log_transmat[0,0], <float*> &log_transmat_T[0,0],
                <float*> &log_startprob[0], <float*> &means[0,0],
                <float*> &vars[0,0], <const float**> seq_pointers,
                self.n_sequences, <int*> &seq_lengths[0],
                self.n_features, self.n_states, 
                <float*> &transcounts[0,0], <float*> &obs[0,0], 
                <float*> &obs2[0,0], <float*> &post[0], &logprob)
        else:
            raise RuntimeError('Invalid precision')

        free(seq_pointers)
        return logprob, {'trans': transcounts, 'obs': obs, 'obs**2': obs2, 'post': post}
