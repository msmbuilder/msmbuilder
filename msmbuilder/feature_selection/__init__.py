# Author: Carlos Xavier Hernandez <cxh@stanford.edu>
# Contributors:
# Copyright (c) 2016, Stanford University and the Authors
# All rights reserved.

from __future__ import print_function, division, absolute_import

from .base import MultiSequenceFeatureSelectionMixin
from .featureselector import FeatureSelector

from sklearn import feature_selection


class GenericUnivariateSelect(MultiSequenceFeatureSelectionMixin,
                              feature_selection.GenericUnivariateSelect):
    __doc__ = feature_selection.GenericUnivariateSelect.__doc__


class RFE(MultiSequenceFeatureSelectionMixin, feature_selection.RFE):
    __doc__ = feature_selection.RFE.__doc__


class RFECV(MultiSequenceFeatureSelectionMixin, feature_selection.RFECV):
    __doc__ = feature_selection.RFECV.__doc__


class SelectFdr(MultiSequenceFeatureSelectionMixin,
                feature_selection.SelectFdr):
    __doc__ = feature_selection.SelectFdr.__doc__


class SelectFpr(MultiSequenceFeatureSelectionMixin,
                feature_selection.SelectFpr):
    __doc__ = feature_selection.SelectFpr.__doc__


class SelectFwe(MultiSequenceFeatureSelectionMixin,
                feature_selection.SelectFwe):
    __doc__ = feature_selection.SelectFwe.__doc__


class SelectKBest(MultiSequenceFeatureSelectionMixin,
                  feature_selection.SelectKBest):
    __doc__ = feature_selection.SelectKBest.__doc__


class SelectPercentile(MultiSequenceFeatureSelectionMixin,
                       feature_selection.SelectPercentile):
    __doc__ = feature_selection.SelectPercentile.__doc__


class VarianceThreshold(MultiSequenceFeatureSelectionMixin,
                        feature_selection.VarianceThreshold):
    __doc__ = feature_selection.VarianceThreshold.__doc__

if hasattr(feature_selection, 'SelectFromModel'):
    class SelectFromModel(MultiSequenceFeatureSelectionMixin,
                          feature_selection.SelectFromModel):
        __doc__ = feature_selection.SelectFromModel.__doc__
