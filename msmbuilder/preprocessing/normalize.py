# Author: Evan N. Feinberg <efein19@gmail.com>
# Contributors: Christian Schwantes <schwancr@gmail.com>m Robert McGibbon <rmcgibbo@gmail.com>, Kyle A. Beauchamp  <kyleabeauchamp@gmail.com>
# Copyright (c) 2016, Stanford University
# All rights reserved.

#-----------------------------------------------------------------------------
# Imports
#-----------------------------------------------------------------------------

from __future__ import print_function, division, absolute_import
import numpy as np
import warnings
from ..base import BaseEstimator
from ..utils import check_iter_of_sequences, array2d
from sklearn.base import TransformerMixin

#-----------------------------------------------------------------------------
# Code
#-----------------------------------------------------------------------------

class Normalize(BaseEstimator, TransformerMixin):
    """Transform data to zero mean and unit variance


    Attributes
    ----------

    means_ : array-like, shape (1, n_features)
        Overall mean of each feature over all data.
    stdevs_ : array-like, shape (1, n_features)
        Overall standard deviation of each feature.
    n_observations_ : int
        Total number of data points fit by the model. Note that the model
        is "reset" by calling `fit()` with new sequences, whereas
        `partial_fit()` updates the fit with new data, and is suitable for
        online learning.
    n_sequences_ : int
        Total number of sequences fit by the model. Note that the model
        is "reset" by calling `fit()` with new sequences, whereas
        `partial_fit()` updates the fit with new data, and is suitable for
         online learning.

    Note: Numerically stable method of computing variance adapted from:
    https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance
    """

    def __init__(self):

        """
        Initialize variables. Variance calculation is simpler with two passes over the data.
            We require passing `sequences` to __init__() such that the total number of data
            points `n` can be calculated here. This is needed for numerically stable 
            calculation 

        """
        self.M = 0.
        self.sum1 = 0.
        self.sum2 = 0.
        self.sum3 = 0.
        
    @property
    def mean(self):
        return self.M
    @property
    def std(self):
        if self.n==1:
            return float('nan')
        return np.sqrt(self.var)

    @property
    def var(self):
        if self.n==1:
            return float('nan')
        return (self.sum2 - np.square(self.sum3) / self.n) / (self.n-1.)

    def fit(self, sequences):
        """Fit the model with a collection of sequences.

        This method is not online.  Any state accumulated from previous calls to
        fit() or partial_fit() will be cleared. For online learning, use
        `partial_fit`.


        Returns
        -------
        self : object
            Returns the instance itself.
        """
        
        check_iter_of_sequences(sequences, max_iter=3)  # we might be lazy-loading

        self.k = len(sequences)
        self.n = 0.
        self.sum1 = 0.
        self.sum2 = 0.
        self.sum3 = 0.

        #First Pass
        for sequence in sequences:
            self.n += float(np.shape(sequence)[0])
            self.sum1 += np.sum(sequence, axis=0)
        self.M = self.sum1 / self.n 

        #Second Pass:
        for X in sequences:
            self._fit(X)

        return self

    def transform_in_place(self, sequences):
        """Apply the dimensionality reduction on X.

        Parameters
        ----------
        sequences: list of arrays holding data

        Returns
        -------
        sequence_new : list of array-like, each of shape (n_samples_i, n_components)

        """

        check_iter_of_sequences(sequences, max_iter=3)  # we might be lazy-loading

        for i, X in enumerate(sequences):
            X = array2d(X)

            X = (X - self.mean) / self.std

            sequences[i] = X


    def fit_transform_in_place(self, sequences):
        """Fit the model with X and apply the dimensionality reduction on X.

        This method is not online. Any state accumulated from previous calls to
        `fit()` or `partial_fit()` will be cleared. For online learning, use
        `partial_fit`.

        Parameters
        ----------
        sequences: list of arrays holding data
        Returns
        -------
        sequence_new : list of array-like, each of shape (n_samples_i, n_components)
        """

        self.fit(sequences)
        self.transform_in_place(sequences)

    def _fit(self,X):
        if X is None:
            return
        
        self.sum2 += np.sum(np.square((X-self.mean)), axis=0)
        self.sum3 += np.sum(X-self.mean, axis=0)

