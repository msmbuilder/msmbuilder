cimport numpy as np


cdef class GaussianHMMCPUImpl:
    cdef list sequences
    cdef int n_sequences
    cdef np.ndarray seq_lengths
    cdef int n_states, n_features
    cdef str precision
    cdef np.ndarray means, vars, log_transmat, log_transmat_T, log_startprob