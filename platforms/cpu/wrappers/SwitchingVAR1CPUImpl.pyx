cimport numpy as np
from GaussianHMMCPUImpl cimport GaussianHMMCPUImpl

cdef extern from "do_estep.hpp" namespace "Mixtape":

    void do_estep_mixed "Mixtape::do_estep<double>"(
        const float* log_transmat, const float* log_transmat_T,
        const float* log_startprob, const float* means,
        const float* variances, const float** sequences,
        const int n_sequences, const np.int32_t* sequence_lengths,
        const int n_features, const int n_states,
        float* transcounts, float* obs, float* obs2,
        float* post, float* logprob) nogil

cdef class SwitchingVAR1CPUImpl(GaussianHMMCPUImpl):
    def __cinit__(self, n_states, n_features):
        self.n_states = n_states
        self.n_features = n_features

    def do_estep(self):
        #starttime = time.time()
        cdef np.ndarray[ndim=2, mode='c', dtype=np.float32_t] log_transmat = self.log_transmat
        cdef np.ndarray[ndim=2, mode='c', dtype=np.float32_t] log_transmat_T = self.log_transmat_T
        cdef np.ndarray[ndim=1, mode='c', dtype=np.float32_t] log_startprob = self.log_startprob
        cdef np.ndarray[ndim=2, mode='c', dtype=np.float32_t] means = self.means
        cdef np.ndarray[ndim=2, mode='c', dtype=np.float32_t] vars = self.vars
        cdef np.ndarray[ndim=1, mode='c', dtype=np.int32_t] seq_lengths = self.seq_lengths

        # All of the sufficient statistics
        cdef np.ndarray[ndim=2, mode='c', dtype=np.float32_t] transcounts = np.zeros((self.n_states, self.n_states), dtype=np.float32)

        cdef np.ndarray[ndim=2, mode='c', dtype=np.float32_t] obs = np.zeros((self.n_states, self.n_features), dtype=np.float32)
        cdef np.ndarray[ndim=2, mode='c', dtype=np.float32_t] obs_but_first = np.zeros((self.n_states, self.n_features), dtype=np.float32)
        cdef np.ndarray[ndim=2, mode='c', dtype=np.float32_t] obs_but_last = np.zeros((self.n_states, self.n_features), dtype=np.float32)

        cdef np.ndarray[ndim=2, mode='c', dtype=np.float32_t] obs_obs_T = np.zeros((self.n_states, self.n_features, self.n_features), dtype=np.float32)
        cdef np.ndarray[ndim=2, mode='c', dtype=np.float32_t] obs_obs_T_offset = np.zeros((self.n_states, self.n_features, self.n_features), dtype=np.float32)
        cdef np.ndarray[ndim=2, mode='c', dtype=np.float32_t] obs_obs_T_but_first = np.zeros((self.n_states, self.n_features, self.n_features), dtype=np.float32)
        cdef np.ndarray[ndim=2, mode='c', dtype=np.float32_t] obs_obs_T_but_last = np.zeros((self.n_states, self.n_features, self.n_features), dtype=np.float32)

        cdef np.ndarray[ndim=1, mode='c', dtype=np.float32_t] post = np.zeros(self.n_states, dtype=np.float32)
        cdef np.ndarray[ndim=1, mode='c', dtype=np.float32_t] post_but_first = np.zeros(self.n_states, dtype=np.float32)
        cdef np.ndarray[ndim=1, mode='c', dtype=np.float32_t] post_but_last = np.zeros(self.n_states, dtype=np.float32)
