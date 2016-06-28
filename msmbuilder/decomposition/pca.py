# Author: Matthew Harrigan <matthew.harrigan@outlook.com>
# Contributors:
# Copyright (c) 2016, Stanford University and the Authors
# All rights reserved.

from __future__ import print_function, division, absolute_import

from sklearn import decomposition

from .base import MultiSequenceDecompositionMixin

__all__ = ['PCA', 'SparsePCA']


class PCA(MultiSequenceDecompositionMixin, decomposition.PCA):
    __doc__ = decomposition.PCA.__doc__

    def summarize(self):
        return '\n'.join([
            "Principal Component Analysis (PCA)",
            "----------",
            "Number of components:    {n_components}",
            "explained variance raio: {explained_variance_ratio_}",
            "Noise variance:          {noise_variance_}",
        ]).format(**self.__dict__)


class SparsePCA(MultiSequenceDecompositionMixin, decomposition.SparsePCA):
    __doc__ = decomposition.SparsePCA.__doc__

    def summarize(self):
        return '\n'.join([
            "Sparse PCA",
            "----------",
            "Number of components:    {n_components}",
        ]).format(**self.__dict__)


class MiniBatchSparsePCA(MultiSequenceDecompositionMixin,
                         decomposition.MiniBatchSparsePCA):
    __doc__ = decomposition.MiniBatchSparsePCA.__doc__

    def summarize(self):
        return '\n'.join([
            "MiniBatch Sparse PCA",
            "--------------------",
            "Number of components:    {n_components}",
            "Batch size:              {batch_size}"
        ]).format(**self.__dict__)


class KernelPCA(MultiSequenceDecompositionMixin, decomposition.KernelPCA):
    __doc__ = decomposition.KernelPCA.__doc__

    def summarize(self):
        return '\n'.join([
            "Kernel PCA",
            "--------------------",
            "Number of components:    {n_components}",
            "Kernel:              {kernel}",
        ]).format(**self.__dict__)
