# Author: Carlos Xavier Hernandez <cxh@stanford.edu>
# Contributors:
# Copyright (c) 2016, Stanford University and the Authors
# All rights reserved.

import numpy as np
from ._mad import mad_estimate

from .base import MultiSequenceOnlinePreprocessingMixin

__all__ = ['MADScaler']


class MADScaler(MultiSequenceOnlinePreprocessingMixin):
    """Scale features using the median absolute deviation.

    This Scaler removes the median and scales the data according to
    the median absolute deviation (MAD). The MAD is the median of the
    absolute residuals (deviations) from the sample median. This scaler uses
    an online estimate of the sample median to accomodate large datasets.

    Standardization of a dataset is a common requirement for many
    machine learning estimators. Typically this is done by removing the mean
    and scaling to unit variance. However, outliers can often influence the
    sample mean / variance in a negative way. In such cases, the median and
    the median absolute deviation often give better results.

    Parameters
    ----------
    eta : float, optional, default=0.001
        Online learning rate.

    References
    ----------
    .. [1] "'On-line' (iterator) algorithms for estimating statistical median, mode, skewness, kurtosis?". StackOverflow. <http://stackoverflow.com/questions/1058813/on-line-iterator-algorithms-for-estimating-statistical-median-mode-skewnes>.
    .. [2] "Median absolute deviation". Wikipedia. <https://en.wikipedia.org/wiki/Median_absolute_deviation>.
    """

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
