import numpy as np
cimport numpy as np
import scipy.stats
np.import_array()

cdef extern double invpsi(double y) nogil
cdef extern int gamma_mixture(const float* X, const int n_samples, const int n_features,
                              const int n_components, int n_iters, double* alpha,
                              double* rate, double* pi) nogil
                 

#print invpsi(3.0)
import numpy as np
import scipy.special
from scipy.misc import logsumexp
 
def invpsi(y):
    """
    Inverse digamma (psi) function.  The digamma function is the
    derivative of the log gamma function.
    """
    # Adapted from matlab code in PMTK (https://code.google.com/p/pmtk3), copyright
    # Kevin Murphy and available under the MIT license.
 
    # Newton iteration to solve digamma(x)-y = 0
    x = np.exp(y) + 0.5
    mask = y < -2.22
    x[mask] = 1.0 / (y[mask] - scipy.special.psi(1))
 
    # never more than 5 iterations required
    for i in range(5):
        x = x - (scipy.special.psi(x)-y) / scipy.special.polygamma(1, x)
    return x


def fit(x, alpha, rate, pi, n_components=3):
    n_samples, n_features = x.shape
    log_x = np.log(x)
    print "N_samples", n_samples
    print "n_components", n_components
    print "n_features", n_features
 
    for i in range(4):
        logg = np.zeros((n_samples, n_components))
        for f in range(n_features):
            # log probability of each data point in each component
            logg += alpha[:, f]*np.log(rate[:, f]) - scipy.special.gammaln(alpha[:, f]) + \
                    np.multiply.outer(log_x[:, f], alpha[:, f]-1) - np.multiply.outer(x[:, f], rate[:, f])
                    
        logp = np.log(pi) + logg - logsumexp(logg, axis=1, b=pi[np.newaxis])[:, np.newaxis]
        p = np.exp(logp)

        # new mixing weights
        pi = np.mean(p, axis=0)

        # new rate and scale parameters
        A = np.einsum('ik,ij->jk', log_x, p)
        B = np.einsum('jk,ij->jk', np.log(rate), p)

        alpha_argument = (A + B) / np.sum(p, axis=0)[:, np.newaxis]
        rate = alpha*np.sum(p, axis=0)[:, np.newaxis] / np.einsum('ik,ij->jk', x, p)
        # when the fit is bad (early iterations), this conditional maximum
        # likelihood update step is not guarenteed to keep alpha positive,
        # which causes the next iteration to be f*cked.
        alpha = np.maximum(invpsi(alpha_argument), 1e-8)
        
        print 'alpha'
        print alpha
        print 'rate'
        print rate
        print 'pi'
        print pi


def main():
    cdef np.ndarray[mode='c', dtype=np.float32_t, ndim=2] X
    cdef np.ndarray[mode='c', dtype=np.double_t, ndim=2] alpha
    cdef np.ndarray[mode='c', dtype=np.double_t, ndim=2] rate
    cdef np.ndarray[mode='c', dtype=np.double_t, ndim=1] pi
    #np.random.seed(42)
    X = np.array(np.concatenate((scipy.stats.distributions.gamma(9,3).rvs(100),
                                 scipy.stats.distributions.gamma(3,0.2).rvs(500))).reshape(-1,1), dtype=np.float32)
    print 'X.shape', X.shape[0], X.shape[1]

    cdef int n_samples = X.shape[0]
    cdef int n_features = X.shape[1]
    cdef int n_components = 3
    alpha = np.asarray((3+np.arange(n_components*n_features)).reshape(n_components, n_features), dtype=np.double)
    rate = np.asarray((10+np.arange(n_components*n_features)).reshape(n_components, n_features), dtype=np.double)
    pi = np.ones(n_components, dtype=np.double)

    gamma_mixture(&X[0,0], n_samples, n_features, n_components, 100, &alpha[0,0], &rate[0,0], &pi[0])

    alpha = np.asarray((3+np.arange(n_components*n_features)).reshape(n_components, n_features), dtype=np.double)
    rate = np.asarray((10+np.arange(n_components*n_features)).reshape(n_components, n_features), dtype=np.double)
    pi = np.ones(n_components, dtype=np.double)
    #fit(X, alpha, rate, pi, n_components)
    
    
main()