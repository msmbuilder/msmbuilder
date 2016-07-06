# Author: Carlos Xavier Hernandez <cxh@stanford.edu>
# Contributors:
# Copyright (c) 2016, Stanford University and the Authors
# All rights reserved.

import numpy as np
from ._mad import mad_estimate

from .base import MultiSequenceOnlinePreprocessingMixin

__all__ = ['MADScaler']


class MADScaler(MultiSequenceOnlinePreprocessingMixin):

    def partial_fit(self, sequence):

        X = np.atleast_2d(sequence).reshape(sequence.shape[0], -1)

        if not hasattr(self, 'medians_'):
            self.medians_ = np.zeros(X.shape[1])
        if not hasattr(self, 'mad_'):
            self.mad_ = np.zeros(X.shape[1])

        self.mad_, self.medians_ = mad_estimate(X, self.mad_, self.medians_,
                                                self.eta)
        return self

    def partial_transform(self, sequence):
        return (sequence - self.medians_) / self.mad_

    def __init__(self, eta=0.001):
        self.eta = eta
