from libc.stdlib cimport malloc, free
import numpy as np
cimport numpy as np

cdef extern from "CUDAGaussianHMM.hpp" namespace "Mixtape":
    cdef cppclass CPPCUDAGaussianHMM "Mixtape::CUDAGaussianHMM":
        CPPCUDAGaussianHMM(const float**, const np.int32_t,
                           const np.int32_t*, const np.int32_t,
                           const np.int32_t)
        void setMeans(float* means)
        void setVariances(float* variances)
        void setTransmat(float* transmat)
        void setStartProb(float* startProb)
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
    cdef int n_sequences
    cdef np.ndarray transmat
    cdef int n_observations, n_states, n_features

    def __cinit__(self, sequences, int n_states):
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
        self.sequences = sequences
        self.n_states = n_states
        self.n_observations = seq_lengths.sum()

        self.thisptr = new CPPCUDAGaussianHMM(
            <const float**>seq_pointers, self.n_sequences, &seq_lengths[0],
            n_states, self.n_features)

        free(seq_pointers)


    def setMeans(self, means):
        cdef np.ndarray[ndim=2, mode='c', dtype=np.float32_t] m = np.asarray(means, order='c', dtype=np.float32)
        if (m.shape[0] != self.n_states) or (m.shape[1] != self.n_features):
            raise TypeError('Shape')
        self.thisptr.setMeans(&m[0,0])

    def setVariances(self, variances):
        cdef np.ndarray[ndim=2, mode='c', dtype=np.float32_t] m = np.asarray(variances, order='c', dtype=np.float32)
        if (m.shape[0] != self.n_states) or (m.shape[1] != self.n_features):
            raise TypeError('Shape')
        self.thisptr.setVariances(&m[0,0])

    def setTransmat(self, transmat):
        cdef np.ndarray[ndim=2, mode='c', dtype=np.float32_t] m = np.asarray(transmat, order='c', dtype=np.float32)
        if (m.shape[0] != self.n_states) or (m.shape[1] != self.n_states):
            raise TypeError('Shape')
        self.transmat = m
        self.thisptr.setTransmat(&m[0,0])

    def setStartprob(self, startprob):
        cdef np.ndarray[ndim=1, mode='c', dtype=np.float32_t] m = np.asarray(startprob, order='c', dtype=np.float32)
        if (m.shape[0] != self.n_states):
            raise TypeError('Shape')
        self.thisptr.setStartProb(&m[0])

    def computeEStep(self):
        self.thisptr.computeEStep()

    def getFrameLogProb(self):
        cdef np.ndarray[ndim=2, dtype=np.float32_t] X = np.zeros((self.n_observations, self.n_states), dtype=np.float32)
        self.thisptr.getFrameLogProb(&X[0,0])
        return X

    def getFwdLattice(self):
        cdef np.ndarray[ndim=2, dtype=np.float32_t] X = np.zeros((self.n_observations, self.n_states), dtype=np.float32)
        self.thisptr.getFwdLattice(&X[0,0])
        return X

    def getBwdLattice(self):
        cdef np.ndarray[ndim=2, dtype=np.float32_t] X = np.zeros((self.n_observations, self.n_states), dtype=np.float32)
        self.thisptr.getBwdLattice(&X[0,0])
        return X

    def getPosteriors(self):
        cdef np.ndarray[ndim=2, dtype=np.float32_t] X = np.zeros((self.n_observations, self.n_states), dtype=np.float32)
        self.thisptr.getPosteriors(&X[0,0])
        return X

    def getSufficientStatistics(self):
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
        print 'Deallocating'
        if self.thisptr is not NULL:
            del self.thisptr

def main():
    np.random.seed(42)
    from pprint import pprint
    means = np.random.rand(4, 2)
    variances = np.ones((4, 2))
    startprob = [0.25, 0.25, 0.25, 0.25]
    transmat = np.random.rand(4,4)
    t1 = [np.random.randn(12,2), 1+np.random.randn(8,2)]
    t2 = [np.concatenate(t1)]

    q1 = CUDAGaussianHMM(t1, 4)
    q2 = CUDAGaussianHMM(t2, 4)
    for q in [q1, q2]:
        q.setMeans(means)
        q.setVariances(variances)
        q.setStartprob(startprob)
        q.setTransmat(transmat)
        q.computeEStep()
        print 'framelogprob'
        print q.getFrameLogProb()


    # obs = q.getSufficientStatistics()
    # print 'cuda'
    # pprint(obs)
    # print 'ref'
    # print 'obs\n', np.dot(q.getPosteriors().T, t[0])
    # print 'obs**2\n', np.dot(q.getPosteriors().T, t[0]**2)
    # print 'post\n', q.getPosteriors().sum(axis=0)
    # print 'trans\n', q._naiveCounts()


main()
