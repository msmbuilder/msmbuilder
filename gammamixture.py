"""
Gamma distribution mixture model
"""
# Author: Robert McGibbon
# Contributors:

import numpy as np
from sklearn import cluster
from sklearn.utils.extmath import logsumexp
import _gammahmm


class GammaMixtureModel(object):
    """Gamma Mixture Model

    Representation of a Gamma mixture model probability distribution.
    This class allows for easy evaluation of, sampling from, and
    maximum-likelihood estimation of the parameters of a Gamma mixture
    distribution.

    Parameters
    ----------
    n_components : int, optional
        Number of mixture components. Defaults to 1.
    random_state: RandomState or an int seed (0 by default)
        A random number generator instance
    n_iter : int, optional
        Number of EM iterations to perform.
    n_init : int, optional
        Number of initializations to perform. The best results is kept

    Attributes
    ----------
    alphas_ : array, shape=[n_components, n_features]
        Shape parameters for the gamma distribution of each state along each
        features
    betas_ : array, shape=[n_components, n_features]
        Rate parameters for the gamma distribution of each state along each
        features. The reciprocal of `beta_` is the scale parameter, sometimes
        notated `theta`.
    weights_ : array, shape=[n_components,]
        The mixing weights for each mixture component.
    """
    def __init__(self, n_components, random_state=None, n_init=1,
                 n_iter=100):
        self.n_components = n_components
        self.random_state = random_state
        self.n_init = n_init
        self.n_iter = n_iter

    def score_samples(self, X):
        """Return the per-sample likelihood of the data under the model.

        Compute the log probability of X under the model and return
        the posterior distribution (responsibilties) of each mixture
        component for each element of X.

        Parameters
        ----------
        X : array_like, shape=[n_samples, n_features]
            List of n_features-dimensional data points.  Each row
            corresponds to a single data point.

        Returns
        -------
        logprob : array, shape=[n_samples,]
            Log probabilities of each data point in X.
        responsibilities : array, shape=[n_samples, n_components]
            Posterior probabilties of each mixture component for each
            observation.
        """
        X = np.asarray(X)
        if X.ndim == 1:
            X = X[:, np.newaxis]
        if X.shape[1] != self.alphas_.shape[1]:
            raise ValueError('The shape of X  is not compatible with self')

        if np.any(X <= 0):
            raise ValueError('All values of X must be positive')
        log_X = np.log(X)


        lpr = np.zeros((len(X), self.n_components))
        for f in range(X.shape[1]):
            # log probability of each data point in each component
            lpr += self.alphas_[:, f]*np.log(self.betas_[:, f]) - scipy.special.gammaln(self.alphas_[:, f]) + \
                   np.multiply.outer(log_X[:, f], self.alphas_[:, f]-1) - np.multiply.outer(X[:, f], self.betas_[:, f])

        lpr += np.log(self.weights_)
        logprob = logsumexp(lpr, axis=1)
        responsibilities = np.exp(lpr - logprob[:, np.newaxis])
        return logprob, responsibilities

    def score(self, X):
        """Compute the log probability under the model.

        Parameters
        ----------
        X : array_like, shape (n_samples, n_features)
            List of n_features-dimensional data points.  Each row
            corresponds to a single data point.

        Returns
        -------
        logprob : array_like, shape (n_samples,)
            Log probabilities of each data point in X
        """
        logprob, _ = self.score_samples(X)
        return logprob


    def predict(self, X):
        """Predict label for data.

        Parameters
        ----------
        X : array-like, shape = [n_samples, n_features]

        Returns
        -------
        C : array, shape = (n_samples,)
        """
        logprob, responsibilities = self.score_samples(X)
        return responsibilities.argmax(axis=1)

    def predict_proba(self, X):
        """Predict posterior probability of data under each Gamma
        distribution in the model in the model.

        Parameters
        ----------
        X : array-like, shape = [n_samples, n_features]

        Returns
        -------
        responsibilities : array-like, shape = (n_samples, n_components)
            Returns the probability of the sample for each Gamma distribution
            (component) in the model.
        """
        logprob, responsibilities = self.score_samples(X)
        return responsibilities

    def sample(self, n_samples=1, random_state=None):
        """Generate random samples from the model.

        Parameters
        ----------
        n_samples : int, optional
            Number of samples to generate. Defaults to 1.

        Returns
        -------
        X : array_like, shape (n_samples, n_features)
            List of samples
        """
        pass

    def fit(self, X):
        """Estimate model parameters with the expectation-maximization
        algorithm.

        A initialization step is performed before entering the em
        algorithm. If you want to avoid this step, set the keyword
        argument init_params to the empty string '' when creating the
        GMM object. Likewise, if you would like just to do an
        initialization, set n_iter=0.

        Parameters
        ----------
        X : array_like, shape (n_samples, n_features)
            List of n_features-dimensional data points.  Each row
            corresponds to a single data point.
        """
        X = np.asarray(X, dtype=np.float32, order='C')
        if X.ndim == 1:
            X = np.asarray(X[:, np.newaxis], order='C')
        if X.shape[0] < self.n_components:
            raise ValueError('Gamma mixture estimation with %s components, '
                             'but only got %s samples'
                             % (self.n_components, X.shape[0]))
        if np.any(X <= 0):
            raise ValueError('All values of X must be positive')
        n_features = X.shape[1]

        for _ in range(self.n_init):
            means = cluster.KMeans(
                n_clusters=self.n_components,
                random_state=self.random_state).fit(X).cluster_centers_
            self.alphas_ = np.ones((self.n_components, n_features))
            self.betas_ = self.alphas_ / means
            self.weights_ = np.ones(self.n_components)

        self.alphas_ = np.asarray(self.alphas_, order='c', dtype=np.double)
        self.betas_ = np.asarray(self.betas_, order='c', dtype=np.double)
        self.weights_ = np.asarray(self.weights_, order='c', dtype=np.double)

        _gammahmm.gamma_mixture_fit(X, self.alphas_, self.betas_, self.weights_,
                                    self.n_iter)

    def bic(self, X):
        """Bayesian information criterion for the current model fit
        and the proposed data

        Parameters
        ----------
        X : array_like, shape (n_samples, n_features)
            List of n_features-dimensional data points.  Each row
            corresponds to a single data point.

        Returns
        -------
        bic: float (the lower the better)
        """
        return (-2 * self.score(X).sum() +
                self._n_parameters() * np.log(X.shape[0]))


    def aic(self, X):
        """Akaike information criterion for the current model fit
        and the proposed data

        Parameters
        ----------
        X : array_like, shape (n_samples, n_features)
            List of n_features-dimensional data points.  Each row
            corresponds to a single data point.

        Returns
        -------
        aic: float (the lower the better)
        """
        return - 2 * self.score(X).sum() + 2 * self._n_parameters()

    def _n_parameters(self):
        """Return the number of free parameters in the model."""
        return self.alphas_.size + self.betas_.size + self.weights_.size

if __name__ == '__main__':
    import scipy.stats
    import itertools

    data = []
    n_features = 3
    for i in range(n_features):
        data.append(np.concatenate((scipy.stats.distributions.gamma(1, (i+1)*20).rvs(1000),
                                    scipy.stats.distributions.gamma(10, (i+1)*20).rvs(3000))))
    data = np.vstack(data).T
    print data.shape
    data = data[np.random.permutation(len(data))]
    test = data[0:len(data)/5]
    train = data[len(data)/5:]

    n_components = range(1,10)
    bics = []
    test_ll = []
    for i in n_components:
        gmm = GammaMixtureModel(n_components=i, n_iter=1000)
        gmm.fit(train)
        bics.append(gmm.bic(train))
        test_ll.append(gmm.score(test).sum())
        print bics
        print test_ll

    import matplotlib.pyplot as pp
    pp.subplot(211)
    pp.plot(n_components, bics, 'x-', c='g', label='bics')
    pp.legend(loc=4)
    pp.gca().twinx().plot(n_components, test_ll, 'x-', c='k', label='testll')
    pp.legend(loc=1)
    pp.xlabel('n states')


    pp.subplot(212)
    colors = itertools.cycle(['r', 'g', 'b', 'c', 'm', 'k'])
    for i in range(n_features):
        pp.hist(data[:, i], bins=15, color=next(colors), alpha=0.3, label='feature %d' % i)
    pp.legend()
    pp.show()
