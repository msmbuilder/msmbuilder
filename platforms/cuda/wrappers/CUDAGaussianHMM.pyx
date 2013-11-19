cimport numpy as np
import numpy as np

cdef extern from "CUDAGaussianHMM.hpp" namespace "Mixtape":
    cdef cppclass CPPCUDAGaussianHMM "Mixtape::CUDAGaussianHMM":
        CPPCUDAGaussianHMM(float*, np.int32_t, np.int32_t*, np.int32_t, np.int32_t)
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
    
cdef class CUDAGaussianHMM:
    cdef CPPCUDAGaussianHMM *thisptr
    cdef np.ndarray trajs
    cdef np.ndarray n_obs
    cdef int n_states, n_features
    
    def __cinit__(self, trajectories, int n_states):
        cdef np.ndarray[ndim=2, dtype=np.float32_t] trajs
        cdef np.ndarray[ndim=1, dtype=np.int32_t] n_obs
        trajs = np.concatenate(trajectories).astype(np.float32)
        n_obs = np.array([len(s) for s in trajectories], dtype=np.int32)

        self.trajs = trajs
        self.n_states = n_states
        self.n_features = trajs.shape[1]
        self.n_obs = n_obs


        self.thisptr = new CPPCUDAGaussianHMM(&trajs[0,0], len(trajectories),
                           &n_obs[0], n_states, self.n_features)


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
        self.thisptr.setTransmat(&m[0,0])

    def setStartprob(self, startprob):
        cdef np.ndarray[ndim=1, mode='c', dtype=np.float32_t] m = np.asarray(startprob, order='c', dtype=np.float32)
        if (m.shape[0] != self.n_states):
            raise TypeError('Shape')
        self.thisptr.setStartProb(&m[0])

    def computeEStep(self):
        self.thisptr.computeEStep()

    def getFrameLogProb(self):
        cdef np.ndarray[ndim=2, dtype=np.float32_t] X = np.zeros((self.n_obs.sum(), self.n_states), dtype=np.float32)
        self.thisptr.getFrameLogProb(&X[0,0])
        return X

    def getFwdLattice(self):
        cdef np.ndarray[ndim=2, dtype=np.float32_t] X = np.zeros((self.n_obs.sum(), self.n_states), dtype=np.float32)
        self.thisptr.getFwdLattice(&X[0,0])
        return X

    def getBwdLattice(self):
        cdef np.ndarray[ndim=2, dtype=np.float32_t] X = np.zeros((self.n_obs.sum(), self.n_states), dtype=np.float32)
        self.thisptr.getBwdLattice(&X[0,0])
        return X

    def getPosteriors(self):
        cdef np.ndarray[ndim=2, dtype=np.float32_t] X = np.zeros((self.n_obs.sum(), self.n_states), dtype=np.float32)
        self.thisptr.getPosteriors(&X[0,0])
        return X

    def getSufficientStatistics(self):
        self.thisptr.initializeSufficientStatistics()
        self.thisptr.computeSufficientStatistics()

        cdef np.ndarray[ndim=2, dtype=np.float32_t] obs = np.zeros((self.n_states, self.n_features), dtype=np.float32)
        cdef np.ndarray[ndim=2, dtype=np.float32_t] obs2 = np.zeros((self.n_states, self.n_features), dtype=np.float32)
        cdef np.ndarray[ndim=1, dtype=np.float32_t] post = np.zeros((self.n_states), dtype=np.float32)

        self.thisptr.getStatsObs(&obs[0,0])
        self.thisptr.getStatsObsSquared(&obs2[0,0])
        self.thisptr.getStatsPost(&post[0])
        
        stats = {'post': post, 'obs': obs, 'obs**2': obs2}
        return stats

    def __dealloc__(self):
        print 'Deallocating'
        if self.thisptr is not NULL:
            del self.thisptr

def main():
    t = [np.random.randn(100,2).astype(np.float32)]
    q = CUDAGaussianHMM(t, 4)
    q.setMeans(10*np.random.rand(4, 2))
    q.setVariances(np.random.rand(4, 2))
    q.setStartprob([0.25, 0.25, 0.25, 0.25])
    q.setTransmat(np.random.rand(4,4))

    q.computeEStep()
    #print q.getFrameLogProb()
    #print q.getFwdLattice()
    obs = q.getSufficientStatistics()
    print 'cuda'
    print obs
    print 'ref'
    print np.dot(q.getPosteriors().T, t[0])
    print np.dot(q.getPosteriors().T, t[0]**2)
    print q.getPosteriors().sum(axis=0)
    print q.getPosteriors()

    

main()
