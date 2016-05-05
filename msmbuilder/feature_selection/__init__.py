# Author: Carlos Xavier Hernandez <cxh@stanford.edu>
# Contributors:
# Copyright (c) 2016, Stanford University and the Authors
# All rights reserved.

from __future__ import print_function, division, absolute_import

from .base import MultiSequenceFeatureSelectionMixin
from .featureselector import FeatureSelector

from sklearn import feature_selection
from sklearn import pipeline


class FeatureUnion(MultiSequenceFeatureSelectionMixin,
                   pipeline.FeatureUnion):
    __doc__ = pipeline.FeatureUnion.__doc__


class RFE(MultiSequenceFeatureSelectionMixin, feature_selection.RFE):
    __doc__ = feature_selection.RFE.__doc__


class RFECV(MultiSequenceFeatureSelectionMixin, feature_selection.RFECV):
    __doc__ = feature_selection.RFECV.__doc__


class VarianceThreshold(MultiSequenceFeatureSelectionMixin,
                        feature_selection.VarianceThreshold):
    __doc__ = feature_selection.VarianceThreshold.__doc__

if hasattr(feature_selection, 'SelectFromModel'):
    class SelectFromModel(MultiSequenceFeatureSelectionMixin,
                          feature_selection.SelectFromModel):
        __doc__ = feature_selection.SelectFromModel.__doc__
