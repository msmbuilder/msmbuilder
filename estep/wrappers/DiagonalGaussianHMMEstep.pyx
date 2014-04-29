import numpy as np
cimport numpy as np
from libc.stdlib cimport malloc, free
np.import_array()

cdef extern from "../include/DiagonalGaussianHMMEstep.hpp":
    cdef _doDiagonalGaussianHMMEstep "Mixtape::doDiagonalGaussianHMMEstep"(
        const int numStates,
        const double* transmat,
        const double* startProb,
        const int numFeatures,
        const double* means,
        const double* variances,
        const int numSequences,
        const int* sequenceLengths,
        const float** sequences)


def doDiagonalGaussianHMMEstep(
        int numStates,
        np.ndarray[ndim=2, mode='c', dtype=np.float64_t] transmat not None,
        np.ndarray[ndim=1, dtype=np.float64_t] startProb not None,
        int numFeatures,
        np.ndarray[ndim=2, dtype=np.float64_t] means not None,
        np.ndarray[ndim=2, dtype=np.float64_t] variances not None,
        sequences):
 
    cdef int numSequences = len(sequences)
    seqPointers = <float**>malloc(numSequences * sizeof(float*))
    sequenceReferences = [0 for i in range(numSequences)]
    cdef np.ndarray[ndim=1, dtype=int] sequenceLengths = np.zeros(numSequences, dtype=np.int32)
    cdef np.ndarray[ndim=2, dtype=np.float32_t] S
    for i in range(numSequences):
        S = np.asarray(sequences[i], order='c', dtype=np.float32)
        sequenceReferences[i] = S
        seqPointers[i] = &S[0,0]
        sequenceLengths[i] = len(S)
        if numFeatures != S.shape[1]:
            raise ValueError('All sequences must be arrays of shape N by %d' % numFeatures)
  

    cdef np.ndarray[ndim=2, mode='c', dtype=np.float32_t] transcounts = np.zeros((numStates, numStates), dtype=np.float32)
    cdef np.ndarray[ndim=2, mode='c', dtype=np.float32_t] obs = np.zeros((numStates, numFeatures), dtype=np.float32)
    cdef np.ndarray[ndim=2, mode='c', dtype=np.float32_t] obs2 = np.zeros((numStates, numFeatures), dtype=np.float32)
    cdef np.ndarray[ndim=1, mode='c', dtype=np.float32_t] post = np.zeros(numStates, dtype=np.float32)

    _doDiagonalGaussianHMMEstep(
        numStates, &transmat[0,0], &startProb[0], numFeatures, &means[0,0], &variances[0,0],
        numSequences, &sequenceLengths[0], <const float**> seqPointers)

    return transcounts, post, obs, obs2
        
        
