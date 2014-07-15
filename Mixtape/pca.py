# Author: Matthew Harrigan <matthew.p.harrigan@gmail.com>
# Contributors:
# Copyright (c) 2014, Stanford University and the Authors
# All rights reserved.
#
# Mixtape is free software: you can redistribute it and/or modify
# it under the terms of the GNU Lesser General Public License as
# published by the Free Software Foundation, either version 2.1
# of the License, or (at your option) any later version.
#
# This library is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU Lesser General Public License for more details.
#
# You should have received a copy of the GNU Lesser General Public
# License along with Mixtape. If not, see <http://www.gnu.org/licenses/>.

from __future__ import print_function, division, absolute_import
from sklearn import decomposition
import numpy as np

from six import PY2

__all__ = ['PCA']


class MultiSequenceDecompositionMixin(object):
    # The API for the scikit-learn decomposition object is, in fit(), that
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

    def fit(self, sequences):
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
        s = super(MultiSequenceDecompositionMixin, self) if PY2 else super()
        s.fit(self._concat(sequences))

        return self

    def _concat(self, sequences):
        self.__lengths = [len(s) for s in sequences]
        if len(sequences) > 0 and isinstance(sequences[0], np.ndarray):
            concat = np.concatenate(sequences)
        else:
            # if the input sequences are not numpy arrays, we need to guess
            # how to concatenate them. this operation below works for mdtraj
            # trajectories (which is the use case that I want to be sure to
            # support), but in general the python container protocol doesn't
            # give us a generic way to make sure we merged sequences
            concat = sequences[0].join(sequences[1:])

        assert sum(self.__lengths) == len(concat)
        return concat

    def _split(self, concat):
        return [concat[cl - l: cl] for (cl, l) in
                zip(np.cumsum(self.__lengths), self.__lengths)]

    def predict(self, sequences):
        """Predict the closest cluster each sample in X belongs to.

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
        s = super(MultiSequenceDecompositionMixin, self) if PY2 else super()
        predictions = []
        for sequence in sequences:
            predictions.append(s.predict(sequence))
        return predictions

    def fit_predict(self, sequences):
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
        s = super(MultiSequenceDecompositionMixin, self) if PY2 else super()
        if hasattr(s, 'fit_predict'):
            labels = s.fit_predict(sequences)
        else:
            self.fit(sequences)
            labels = self.predict(sequences)

        if not isinstance(labels, list):
            labels = self._split(labels)
        return labels

    def transform(self, sequences):
        """Alias for predict"""
        return self.predict(sequences)

    def fit_transform(self, sequences):
        """Alias for fit_predict"""
        return self.fit_predict(sequences)


class PCA(MultiSequenceDecompositionMixin, decomposition.PCA):
    pass
