from libc.stdlib cimport malloc, free
import numpy as np
cimport numpy as np


cdef extern from "CUDAGaussianHMM.hpp" namespace "Mixtape":
    cdef cppclass CPPCUDAGaussianHMM "Mixtape::CUDAGaussianHMM":
        CPPCUDAGaussianHMM(const float**, const np.int32_t,
                           const np.int32_t*, const np.int32_t,
                           const np.int32_t) except +
        void setMeans(const float* means)
        void getMeans(float* means)
        void setVariances(const float* variances)
        void getVariances(float* variances)
        void setTransmat(const float* transmat)
        void getTransmat(float* transmat)
        void setStartProb(const float* startProb)
        void getStartProb(float* startProb)
        
        float computeEStep()
        void initializeSufficientStatistics()
        void computeSufficientStatistics()
        void getFrameLogProb(float* out)
        void getFwdLattice(float* out)
        void getBwdLattice(float* out)
        void getPosteriors(float* out)
        void getStatsObs(float* out)
        void getStatsObsSquared(float* out)
        void getStatsPost(float* out)
        void getStatsTransCounts(float* out)


cdef class CUDAGaussianHMM:
    cdef CPPCUDAGaussianHMM *thisptr
    cdef list sequences
    cdef int n_sequences, n_observations, n_states, n_features

    def __cinit__(self, sequences, int n_states):
        self.sequences = sequences
        self.n_sequences = len(sequences)
        if self.n_sequences <= 0:
            raise ValueError('More than 0 sequences must be provided')
        cdef float** seq_pointers = \
             <float**>malloc(self.n_sequences * sizeof(float*))
        cdef np.ndarray[ndim=1, dtype=np.int32_t] seq_lengths = np.zeros(self.n_sequences, dtype=np.int32)

        cdef np.ndarray[ndim=2, dtype=np.float32_t] S
        for i in range(self.n_sequences):
            sequences[i] = np.asarray(sequences[i], order='c',
                                      dtype=np.float32)
            S = sequences[i]
            seq_pointers[i] = &S[0,0]
            seq_lengths[i] = len(sequences[i])
            if i == 0:
                self.n_features = sequences[i].shape[1]
            else:
                if self.n_features != sequences[i].shape[1]:
                    raise ValueError('All sequences must have the same '
                                     'number of features')

        self.n_states = n_states
        self.n_observations = seq_lengths.sum()

        self.thisptr = new CPPCUDAGaussianHMM(
            <const float**>seq_pointers, self.n_sequences, &seq_lengths[0],
            n_states, self.n_features)

        free(seq_pointers)

    property means_:
        def __get__(self):
            cdef np.ndarray[ndim=2, mode='c', dtype=np.float32_t] x = np.zeros((self.n_states, self.n_features), dtype=np.float32)
            self.thisptr.getMeans(&x[0,0])
            return x
    
        def __set__(self, value):
            cdef np.ndarray[ndim=2, mode='c', dtype=np.float32_t] m = np.asarray(value, order='c', dtype=np.float32)
            if (m.shape[0] != self.n_states) or (m.shape[1] != self.n_features):
                raise TypeError('Means must have shape (%d, %d), You supplied (%d, %d)',
                                (self.n_states, self.n_features, m.shape[0], m.shape[1]))
            self.thisptr.setMeans(&m[0,0])

    property variances_:
        def __get__(self):
            cdef np.ndarray[ndim=2, mode='c', dtype=np.float32_t] v = np.zeros((self.n_states, self.n_features), dtype=np.float32)
            self.thisptr.getVariances(&v[0,0])
            return v

        def __set__(self, value):
            cdef np.ndarray[ndim=2, mode='c', dtype=np.float32_t] v = np.asarray(value, order='c', dtype=np.float32)
            if (v.shape[0] != self.n_states) or (v.shape[1] != self.n_features):
                raise TypeError('Variances must have shape (%d, %d), You supplied (%d, %d)',
                                (self.n_states, self.n_features, v.shape[0], v.shape[1]))
            self.thisptr.setVariances(&v[0,0])

    property transmat_:
        def __get__(self):
            cdef np.ndarray[ndim=2, mode='c', dtype=np.float32_t] t = np.zeros((self.n_states, self.n_states), dtype=np.float32)
            self.thisptr.getTransmat(&t[0,0])
            return t

        def __set__(self, value):
            cdef np.ndarray[ndim=2, mode='c', dtype=np.float32_t] t = np.asarray(value, order='c', dtype=np.float32)
            if (t.shape[0] != self.n_states) or (t.shape[1] != self.n_states):
                raise TypeError('transmat must have shape (%d, %d), You supplied (%d, %d)',
                                (self.n_states, self.n_states, t.shape[0], t.shape[1]))
            self.thisptr.setTransmat(&t[0,0])

    property startprob_:
        def __get__(self):
            cdef np.ndarray[ndim=1, mode='c', dtype=np.float32_t] s = np.zeros(self.n_states, dtype=np.float32)
            s.fill(-123)
            self.thisptr.getStartProb(&s[0])
            return s
    
        def __set__(self, value):
            cdef np.ndarray[ndim=1, mode='c', dtype=np.float32_t] s = np.asarray(value, order='c', dtype=np.float32)
            if (s.shape[0] != self.n_states):
                raise TypeError('startprob must have shape (%d,), You supplied (%d,)',
                                (self.n_states, s.shape[0]))
            self.thisptr.setStartProb(&s[0])

    def _do_estep(self):
        self.thisptr.computeEStep()

    def _get_framelogprob(self):
        cdef np.ndarray[ndim=2, dtype=np.float32_t] X = np.zeros((self.n_observations, self.n_states), dtype=np.float32)
        self.thisptr.getFrameLogProb(&X[0,0])
        return X

    def _get_fwdlattice(self):
        cdef np.ndarray[ndim=2, dtype=np.float32_t] X = np.zeros((self.n_observations, self.n_states), dtype=np.float32)
        self.thisptr.getFwdLattice(&X[0,0])
        return X

    def _get_bwdlattice(self):
        cdef np.ndarray[ndim=2, dtype=np.float32_t] X = np.zeros((self.n_observations, self.n_states), dtype=np.float32)
        self.thisptr.getBwdLattice(&X[0,0])
        return X

    def _get_posteriors(self):
        cdef np.ndarray[ndim=2, dtype=np.float32_t] X = np.zeros((self.n_observations, self.n_states), dtype=np.float32)
        self.thisptr.getPosteriors(&X[0,0])
        return X

    def _get_sufficient_statistics(self):
        self.thisptr.initializeSufficientStatistics()
        self.thisptr.computeSufficientStatistics()

        cdef np.ndarray[ndim=2, dtype=np.float32_t] obs = np.zeros((self.n_states, self.n_features), dtype=np.float32)
        cdef np.ndarray[ndim=2, dtype=np.float32_t] obs2 = np.zeros((self.n_states, self.n_features), dtype=np.float32)
        cdef np.ndarray[ndim=1, dtype=np.float32_t] post = np.zeros((self.n_states), dtype=np.float32)
        cdef np.ndarray[ndim=2, dtype=np.float32_t] trans = np.zeros((self.n_states, self.n_states), dtype=np.float32)

        self.thisptr.getStatsObs(&obs[0,0])
        self.thisptr.getStatsObsSquared(&obs2[0,0])
        self.thisptr.getStatsPost(&post[0])
        self.thisptr.getStatsTransCounts(&trans[0,0])

        stats = {'post': post, 'obs': obs, 'obs**2': obs2, 'trans': trans}
        return stats

    def _naiveCounts(self):
        from scipy.misc import logsumexp
        fwdlattice = self.getFwdLattice()
        bwdlattice = self.getBwdLattice()
        framelogprob = self.getFrameLogProb()
        log_transmat = np.log(self.transmat)
        lnP = logsumexp(fwdlattice[-1])
        lneta = np.zeros((self.n_observations - 1, self.n_states, self.n_states))
        for i in range(self.n_states):
            for j in range(self.n_states):
                for t in range(self.n_observations - 1):
                    lneta[t, i, j] = fwdlattice[t, i] + log_transmat[i, j] \
                                     + framelogprob[t + 1, j] + bwdlattice[t + 1, j] - lnP

        return np.exp(logsumexp(lneta, axis=0))

    def __dealloc__(self):
        if self.thisptr is not NULL:
            del self.thisptr

