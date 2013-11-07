import numpy as np
cimport numpy as np
import scipy.stats
np.import_array()

cdef extern int gamma_mixture(const float* X, const int n_samples,
                              const int n_features, const int n_components,
                              int n_iters, double* alpha,
                              double* rate, double* pi) nogil

def gamma_mixture_fit(
        np.ndarray[dtype=np.float32_t, ndim=2, mode='c'] X not None,
        np.ndarray[dtype=np.double_t, ndim=2, mode='c'] shape not None,
        np.ndarray[dtype=np.double_t, ndim=2, mode='c'] rate not None,
        np.ndarray[dtype=np.double_t, ndim=1, mode='c'] pi not None,
        int n_iters):
    cdef int n_samples = X.shape[0]
    cdef int n_features = X.shape[1]
    cdef int n_components = shape.shape[0]
    assert rate.shape[0] == n_components
    assert shape.shape[1] == n_features
    assert rate.shape[1] == n_features
    gamma_mixture(&X[0,0], n_samples, n_features, n_components, n_iters,
                  &shape[0,0], &rate[0,0], &pi[0])
