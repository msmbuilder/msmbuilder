"""
Gamma distribution mixture model
"""
# Author: Robert McGibbon <rmcgibbo@gmail.com>
# Contributors:
# Copyright (c) 2013, Stanford University
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without modification,
# are permitted provided that the following conditions are met:
#
#   Redistributions of source code must retain the above copyright notice, this
#   list of conditions and the following disclaimer.
#
#   Redistributions in binary form must reproduce the above copyright notice, this
#   list of conditions and the following disclaimer in the documentation and/or
#   other materials provided with the distribution.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
# ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
# WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR
# ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
# (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
# LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON
# ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
# SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

import numpy as np
from scipy.special import gammaln
from sklearn import cluster
from sklearn.utils.extmath import logsumexp
from mixtape import _gamma


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
            lpr += (self.alphas_[:, f]*np.log(self.betas_[:, f])
                    - gammaln(self.alphas_[:, f])
                    + np.multiply.outer(log_X[:, f], self.alphas_[:, f]-1)
                    - np.multiply.outer(X[:, f], self.betas_[:, f]))

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
        raise NotImplementedError()

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

        _gamma.gamma_mixture_fit(X, self.alphas_, self.betas_, self.weights_,
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
