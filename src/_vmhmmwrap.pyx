import cython
import numpy as np
cimport numpy as np
np.import_array()

cdef extern int fitinvkappa(long n_samples, long n_features, long n_components,
                 double* posteriors, double* obs, double* means, double* out) nogil

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


def _compute_log_likelihood_1(obs, means, kappas):
    from scipy.stats.distributions import vonmises
    n_components = kappas.shape[0]
    value = np.array([np.sum(vonmises.logpdf(obs, kappas[i], means[i]), axis=1) for i in range(n_components)]).T
    return value

def _compute_log_likelihood_2(obs, means, kappas):
    return obs


obs = np.random.rand(10,3)
means = np.random.rand(2,3)
kappas = np.random.rand(2,3)
A = _compute_log_likelihood_1(obs, means, kappas)
B = _compute_log_likelihood_2(obs, means, kappas)

np.testing.assert_array_almost_equal(A, B)
exit(1)
