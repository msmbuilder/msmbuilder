import cython
import numpy as np
cimport numpy as np
np.import_array()

cdef extern int fitinvkappa(long n_samples, long n_features, long n_components,
                 double* posteriors, double* obs, double* means, double* out) nogil

cdef extern int  compute_log_likelihood(double* obs, double* means, double* kappas,
                                        long n_samples, long n_components, long n_features,
                                        double* out) nogil


@cython.boundscheck(False)
@cython.wraparound(False)
def _fitinvkappa(np.ndarray[np.double_t, ndim=2, mode="c"] posteriors not None,
                np.ndarray[np.double_t, ndim=2, mode="c"] obs not None,
                np.ndarray[np.double_t, ndim=2, mode="c"] means not None,
                np.ndarray[np.double_t, ndim=2, mode="c"] out not None):
    cdef long n_samples = posteriors.shape[0]
    cdef long n_features = obs.shape[1]
    cdef long n_components = means.shape[0]

    fitinvkappa(n_samples, n_features, n_components, &posteriors[0, 0],
                &obs[0,0], &means[0, 0], &out[0, 0])
    return 1;


@cython.boundscheck(False)
@cython.wraparound(False)
def _compute_log_likelihood(np.ndarray[dtype=np.double_t, ndim=2, mode='c'] obs not None, 
                           np.ndarray[dtype=np.double_t, ndim=2, mode='c'] means not None, 
                           np.ndarray[dtype=np.double_t, ndim=2, mode='c'] kappas not None):
    cdef int n_samples = obs.shape[0]        # large
    cdef int n_components = means.shape[0]   # moderate
    cdef int n_features = means.shape[1]     # small
    cdef np.ndarray[dtype=np.double_t, ndim=2, mode='c'] out
    out = np.empty((n_samples, n_components))

    compute_log_likelihood(&obs[0,0], &means[0,0], &kappas[0,0], n_samples,
                           n_components, n_features, &out[0,0])

    return out
