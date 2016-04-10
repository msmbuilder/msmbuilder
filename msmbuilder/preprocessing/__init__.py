# Author: Carlos Xavier Hernandez <cxh@stanford.edu>
# Contributors:
# Copyright (c) 2016, Stanford University and the Authors
# All rights reserved.

from __future__ import print_function, division, absolute_import

from sklearn import preprocessing

from .base import (MultiSequencePreprocessingMixin,
                   MultiSequenceOnlinePreprocessingMixin)


class Binarizer(MultiSequencePreprocessingMixin, preprocessing.Binarizer):
    __doc__ = preprocessing.Binarizer.__doc__


try:
    class FunctionTransformer(MultiSequencePreprocessingMixin,
                              preprocessing.FunctionTransformer):
        __doc__ = preprocessing.FunctionTransformer.__doc__
except AttributeError:
    pass


class Imputer(MultiSequencePreprocessingMixin, preprocessing.Imputer):
    __doc__ = preprocessing.Imputer.__doc__


class KernelCenterer(MultiSequencePreprocessingMixin,
                     preprocessing.KernelCenterer):
    __doc__ = preprocessing.KernelCenterer.__doc__


class LabelBinarizer(MultiSequencePreprocessingMixin,
                     preprocessing.LabelBinarizer):
    __doc__ = preprocessing.LabelBinarizer.__doc__


class MultiLabelBinarizer(MultiSequencePreprocessingMixin,
                          preprocessing.MultiLabelBinarizer):
    __doc__ = preprocessing.MultiLabelBinarizer.__doc__


class MinMaxScaler(MultiSequenceOnlinePreprocessingMixin,
                   preprocessing.MinMaxScaler):
    __doc__ = preprocessing.MinMaxScaler.__doc__

try:
    class MaxAbsScaler(MultiSequenceOnlinePreprocessingMixin,
                       preprocessing.MaxAbsScaler):
        __doc__ = preprocessing.MaxAbsScaler.__doc__
except AttributeError:
    pass


class Normalizer(MultiSequencePreprocessingMixin, preprocessing.Normalizer):
    __doc__ = preprocessing.Normalizer.__doc__


try:
    class RobustScaler(MultiSequencePreprocessingMixin,
                       preprocessing.RobustScaler):
        __doc__ = preprocessing.RobustScaler.__doc__
except AttributeError:
    pass


class StandardScaler(MultiSequenceOnlinePreprocessingMixin,
                     preprocessing.StandardScaler):
    __doc__ = preprocessing.StandardScaler.__doc__


class PolynomialFeatures(MultiSequencePreprocessingMixin,
                         preprocessing.PolynomialFeatures):
    __doc__ = preprocessing.PolynomialFeatures.__doc__
