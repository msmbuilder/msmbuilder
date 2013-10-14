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
    import scipy.special
    n_samples = obs.shape[0]
    n_components = means.shape[0]
    n_features = means.shape[1]

    result = np.zeros((n_samples, n_components))

    for i in range(n_components):
        for j in range(n_features):
            log_denom = 2*np.pi*scipy.special.iv(0, kappas[i,j])
            
            for k in range(n_samples):
                log_num = np.log(kappas[i, j]*np.cos(obs[k, j] - means[i,j]))
                
                results[i, k] += (log_num - log_denom)

    return result
                       

obs = np.random.rand(10,3)
means = np.random.rand(2,3)
kappas = np.random.rand(2,3)
A = _compute_log_likelihood_1(obs, means, kappas)
B = _compute_log_likelihood_2(obs, means, kappas)
print 'A\n', A, A.shape
print 'B\n', B, B.shape

np.testing.assert_array_almost_equal(A, B)
exit(1)
