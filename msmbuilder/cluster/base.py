# Author: Robert McGibbon <rmcgibbo@gmail.com>
# Contributors:
# Copyright (c) 2014, Stanford University
# All rights reserved.

#-----------------------------------------------------------------------------
# Imports
#-----------------------------------------------------------------------------

from __future__ import absolute_import, print_function, division
import numpy as np

import mdtraj as md
from ..utils import check_iter_of_sequences


class MultiSequenceClusterMixin(object):

    # The API for the scikit-learn Cluster object is, in fit(), that
    # they take a single 2D array of shape (n_data_points, n_features).
    #
    # For clustering a collection of timeseries, we need to preserve
    # the structure of which data_point came from which sequence. If
    # we concatenate the sequences together, we lose that information.
    #
    # This mixin is basically a little "adaptor" that changes fit()
    # so that it accepts a list of sequences. Its implementation
    # concatenates the sequences, calls the superclass fit(), and
    # then splits the labels_ back into the sequenced form.

    _allow_trajectory = False

    def fit(self, sequences, y=None):
        """Fit the  clustering on the data

        Parameters
        ----------
        sequences : list of array-like, each of shape [sequence_length, n_features]
            A list of multivariate timeseries. Each sequence may have
            a different length, but they all must have the same number
            of features.

        Returns
        -------
        self
        """
        check_iter_of_sequences(sequences, allow_trajectory=self._allow_trajectory)
        super(MultiSequenceClusterMixin, self).fit(self._concat(sequences))

        if hasattr(self, 'labels_'):
            self.labels_ = self._split(self.labels_)

        return self

    def _concat(self, sequences):
        self.__lengths = [len(s) for s in sequences]
        if len(sequences) > 0 and isinstance(sequences[0], np.ndarray):
            concat = np.ascontiguousarray(np.concatenate(sequences))
        elif isinstance(sequences[0], md.Trajectory):
            # if the input sequences are not numpy arrays, we need to guess
            # how to concatenate them. this operation below works for mdtraj
            # trajectories (which is the use case that I want to be sure to
            # support), but in general the python container protocol doesn't
            # give us a generic way to make sure we merged sequences
            concat = sequences[:][0]
            if len(sequences) > 1:
                concat = concat.join(sequences[:][1:])
            concat.center_coordinates()
        else:
            raise TypeError('sequences must be a list of numpy arrays '
                            'or ``md.Trajectory``s')

        assert sum(self.__lengths) == len(concat)
        return concat

    def _split(self, concat):
        return [concat[cl - l: cl] for (cl, l) in zip(np.cumsum(self.__lengths), self.__lengths)]

    def _split_indices(self, concat_inds):
        """Take indices in 'concatenated space' and return as pairs
        of (traj_i, frame_i)
        """
        clengths = np.append([0], np.cumsum(self.__lengths))
        mapping = np.zeros((clengths[-1], 2), dtype=int)
        for traj_i, (start, end) in enumerate(zip(clengths[:-1], clengths[1:])):
            mapping[start:end, 0] = traj_i
            mapping[start:end, 1] = np.arange(end - start)
        return mapping[concat_inds]

    def predict(self, sequences, y=None):
        """Predict the closest cluster each sample in each sequence in
        sequences belongs to.

        In the vector quantization literature, `cluster_centers_` is called
        the code book and each value returned by `predict` is the index of
        the closest code in the code book.

        Parameters
        ----------
        sequences : list of array-like, each of shape [sequence_length, n_features]
            A list of multivariate timeseries. Each sequence may have
            a different length, but they all must have the same number
            of features.

        Returns
        -------
        Y : list of arrays, each of shape [sequence_length,]
            Index of the closest center each sample belongs to.
        """
        predictions = []
        check_iter_of_sequences(sequences, allow_trajectory=self._allow_trajectory)
        for X in sequences:
            predictions.append(self.partial_predict(X))
        return predictions

    def partial_predict(self, X, y=None):
        """Predict the closest cluster each sample in X belongs to.

        In the vector quantization literature, `cluster_centers_` is called
        the code book and each value returned by `predict` is the index of
        the closest code in the code book.

        Parameters
        ----------
        X : array-like shape=(n_samples, n_features)
            A single timeseries.

        Returns
        -------
        Y : array, shape=(n_samples,)
            Index of the cluster that each sample belongs to
        """
        if isinstance(X, md.Trajectory):
            X.center_coordinates()
        return super(MultiSequenceClusterMixin, self).predict(X)

    def fit_predict(self, sequences, y=None):
        """Performs clustering on X and returns cluster labels.

        Parameters
        ----------
        sequences : list of array-like, each of shape [sequence_length, n_features]
            A list of multivariate timeseries. Each sequence may have
            a different length, but they all must have the same number
            of features.

        Returns
        -------
        Y : list of ndarray, each of shape [sequence_length, ]
            Cluster labels
        """
        if hasattr(super(MultiSequenceClusterMixin, self), 'fit_predict'):
            check_iter_of_sequences(sequences, allow_trajectory=self._allow_trajectory)
            labels = super(MultiSequenceClusterMixin, self).fit_predict(sequences)
        else:
            self.fit(sequences)
            labels = self.predict(sequences)

        if not isinstance(labels, list):
            labels = self._split(labels)
        return labels

    def transform(self, sequences):
        """Alias for predict"""
        return self.predict(sequences)

    def partial_transform(self, X):
        """Alias for partial_predict"""
        return self.partial_predict(X)

    def fit_transform(self, sequences, y=None):
        """Alias for fit_predict"""
        return self.fit_predict(sequences, y)
