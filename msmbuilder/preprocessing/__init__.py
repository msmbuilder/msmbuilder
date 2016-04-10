# Author: Carlos Xavier Hernandez <cxh@stanford.edu>
# Contributors:
# Copyright (c) 2016, Stanford University and the Authors
# All rights reserved.

from __future__ import print_function, division, absolute_import

from sklearn import preprocessing

from .base import MultiSequencePreprocessingMixin


class Binarizer(MultiSequencePreprocessingMixin, preprocessing.Binarizer):
    __doc__ = preprocessing.Binarizer.__doc__


class FunctionTransformer(MultiSequencePreprocessingMixin,
                          preprocessing.FunctionTransformer):
    __doc__ = preprocessing.FunctionTransformer.__doc__


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


class MinMaxScaler(MultiSequencePreprocessingMixin,
                   preprocessing.MinMaxScaler):
    __doc__ = preprocessing.MinMaxScaler.__doc__


class MaxAbsScaler(MultiSequencePreprocessingMixin,
                   preprocessing.MaxAbsScaler):
    __doc__ = preprocessing.MaxAbsScaler.__doc__


class Normalizer(MultiSequencePreprocessingMixin, preprocessing.Normalizer):
    __doc__ = preprocessing.Normalizer.__doc__


class RobustScaler(MultiSequencePreprocessingMixin,
                   preprocessing.RobustScaler):
    __doc__ = preprocessing.RobustScaler.__doc__


class StandardScaler(MultiSequencePreprocessingMixin,
                     preprocessing.StandardScaler):
    __doc__ = preprocessing.StandardScaler.__doc__


class PolynomialFeatures(MultiSequencePreprocessingMixin,
                         preprocessing.PolynomialFeatures):
    __doc__ = preprocessing.PolynomialFeatures.__doc__
