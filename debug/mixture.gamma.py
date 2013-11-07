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

def fit_multi(x, n_components=3):
    n_samples, n_features = x.shape
    alpha = 10*np.random.rand(n_components, n_features)
    rate = 10*np.random.rand(n_components, n_features)
    pi = np.ones(n_components)
    log_x = np.log(x)
 
    for i in range(50):
        logg = np.zeros((n_samples, n_components))
        for f in range(n_features):
            # log probability of each data point in each component
            logg += alpha[:, f]*np.log(rate[:, f]) - scipy.special.gammaln(alpha[:, f]) + \
                    np.multiply.outer(log_x[:, f], alpha[:, f]-1) - np.multiply.outer(x[:, f], rate[:, f])
        
        logp = np.log(pi) + logg - logsumexp(logg, axis=1, b=pi[np.newaxis])[:, np.newaxis]
        p = np.exp(logp)

        # new mixing weights
        pi = np.mean(np.exp(logp), axis=0)

        # new rate and scale parameters
        A = np.einsum('ik,ij->jk', log_x, p)
        B = np.einsum('jk,ij->jk', np.log(rate), p)
        alpha_argument = (A + B) / np.sum(p, axis=0)[:, np.newaxis]
        rate = alpha*np.sum(p, axis=0)[:, np.newaxis] / np.einsum('ik,ij->jk', x, p)
        # when the fit is bad (early iterations), this conditional maximum
        # likelihood update step is not guarenteed to keep alpha positive,
        # which causes the next iteration to be f*cked.
        alpha = np.maximum(invpsi(alpha_argument), 1e-8)
    
    return alpha, rate, pi


def fit(x, k=3):
    alpha = 10*np.random.rand(k)
    rate = 10*np.random.rand(k)
    pi = np.ones(k)
    log_x = np.log(x)
 
    for i in range(50):
        # log probability of each data point in each component
        logg = alpha*np.log(rate) - scipy.special.gammaln(alpha) + \
            np.multiply.outer(log_x, alpha-1) - np.multiply.outer(x, rate)
        logp = np.log(pi) + logg - logsumexp(logg, axis=1, b=pi[np.newaxis])[:, np.newaxis]
        p = np.exp(logp)

        # new mixing weights
        pi = np.mean(np.exp(logp), axis=0)

        # new rate and scale parameters
        A = np.einsum('i,ij->j', log_x, p)
        B = np.einsum('j,ij->j', np.log(rate), p)
        alpha_argument = (A + B) / np.sum(p, axis=0)
        rate = alpha*np.sum(p, axis=0) / np.einsum('i,ij->j', x, p)
        # when the fit is bad (early iterations), this conditional maximum
        # likelihood update step is not guarenteed to keep alpha positive,
        # which causes the next iteration to be f*cked.
        alpha = np.maximum(invpsi(alpha_argument), 1e-8)

    return alpha, rate, pi


import matplotlib.pyplot as pp
import scipy.stats
x = np.concatenate((scipy.stats.distributions.gamma(9,3).rvs(500),
                    scipy.stats.distributions.gamma(3,0.2).rvs(500)))
pp.hist(x, bins=50, alpha=0.3)

alpha, rate, pi = fit(x, k=2)
print 'alpha\n', alpha
print 'rate\n', rate
print 'pi\n', pi

xx = np.linspace(0.001, np.max(x), 1000)
g = (np.power(rate, alpha)/scipy.special.gamma(alpha)) * np.power.outer(xx, alpha-1) *  np.exp(-np.multiply.outer(xx, rate))
ax2 = pp.gca().twinx()
ax2.plot(xx, g[:, 0])
ax2.plot(xx, g[:, 1])

print 'samples. features', np.vstack((x, x)).T.shape
alpha, rate, pi = fit_multi(np.vstack((x, x)).T, n_components=3)
print '\n\nalpha\n', alpha
print 'rate\n', rate
print 'pi\n', pi
g = (np.power(rate[:, 0], alpha[:, 0])/scipy.special.gamma(alpha[:, 0])) * np.power.outer(xx, alpha[:, 0]-1) *  np.exp(-np.multiply.outer(xx, rate[:, 0]))
ax2.plot(xx, g[:, 0], 'r')
ax2.plot(xx, g[:, 1], 'r')
ax2.plot(xx, g[:, 2], 'r')

pp.show()


