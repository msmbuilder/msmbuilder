# Author: Carlos Xavier Hernandez <cxh@stanford.edu>
# Contributors:
# Copyright (c) 2016, Stanford University and the Authors
# All rights reserved.

from __future__ import absolute_import

from .base import MultiSequenceFeatureSelectionMixin
from .featureselector import FeatureSelector

from sklearn import feature_selection

class VarianceThreshold(MultiSequenceFeatureSelectionMixin,
                        feature_selection.VarianceThreshold):
    __doc__ = feature_selection.VarianceThreshold.__doc__
