# Author: Robert McGibbon <rmcgibbo@gmail.com>
# Contributors:
# Copyright (c) 2013, Stanford University
# All rights reserved.

# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are
# met:
#
#   Redistributions of source code must retain the above copyright notice,
#   this list of conditions and the following disclaimer.
#
#   Redistributions in binary form must reproduce the above copyright
#   notice, this list of conditions and the following disclaimer in the
#   documentation and/or other materials provided with the distribution.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS
# IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED
# TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A
# PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
# HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
# SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED
# TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
# PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
# LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
# NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
# SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

#-----------------------------------------------------------------------------
# Imports
#-----------------------------------------------------------------------------

from __future__ import absolute_import, print_function, division
from six import PY2
import numpy as np
from sklearn import cluster

__all__ = ['KMeans', 'MiniBatchKMeans', 'AffinityPropagation', 'MeanShift',
           'SpectralClustering', 'Ward', 'KCenters', 'MultiSequenceClusterMixin']

#-----------------------------------------------------------------------------
# Code
#-----------------------------------------------------------------------------

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

    def fit(self, sequences, y=None):
        concat = np.concatenate(sequences)
        lengths = [len(s) for s in sequences]
        s = super(MultiSequenceClusterMixin, self) if PY2 else super()
        s.fit(concat)
        self.labels_ = self._split(self.labels_, lengths)
        return self
    
    @staticmethod
    def _split(longlist, lengths):
        return [longlist[cl - l: cl] for (cl, l) in zip(np.cumsum(lengths), lengths)]

    def transform(self, sequences):
        s = super(MultiSequenceClusterMixin, self) if PY2 else super()
        transformed = []
        for sequence in sequences:
            transformed.append(s.transform(sequence))
        return transformed
    
    def fit_transform(self, sequences):
        return self.fit(sequences).transform(sequences)

    def predict(self, sequences):
        s = super(MultiSequenceClusterMixin, self) if PY2 else super()
        predictions = []
        for sequence in sequences:
            predictions.append(s.predict(sequence))
        return transformed

#-----------------------------------------------------------------------------
# New "multisequence" versions of all of the clustering algorithims in sklearn
#-----------------------------------------------------------------------------

class KMeans(MultiSequenceClusterMixin, cluster.KMeans):
    pass

class MiniBatchKMeans(MultiSequenceClusterMixin, cluster.MiniBatchKMeans):
    pass

class AffinityPropagation(MultiSequenceClusterMixin, cluster.AffinityPropagation):
    pass

class MeanShift(MultiSequenceClusterMixin, cluster.MeanShift):
    pass
    
class SpectralClustering(MultiSequenceClusterMixin, cluster.SpectralClustering):
    pass

class Ward(MultiSequenceClusterMixin, cluster.Ward):
    pass

# This needs to come _after_ MultiSequenceClusterMixin is defined, to avoid
# recursive circular imports
from .kcenters import KCenters
