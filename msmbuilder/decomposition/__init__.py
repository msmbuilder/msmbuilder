from __future__ import absolute_import

from sklearn import decomposition as _decomposition

from .base import MultiSequenceDecompositionMixin
from .ktica import KernelTICA
from .pca import PCA, SparsePCA, MiniBatchSparsePCA
from .sparsetica import SparseTICA
from .tica import tICA


class FastICA(MultiSequenceDecompositionMixin, _decomposition.FastICA):
    __doc__ = _decomposition.FastICA.__doc__

    def summarize(self):
        return '\n'.join([
            "Independent Component Analysis (ICA)",
            "----------",
            "Number of components:    {n_components}",
            "Number of iterations: {n_iter_}",
        ]).format(**self.__dict__)


class FactorAnalysis(MultiSequenceDecompositionMixin,
                     _decomposition.FactorAnalysis):
    __doc__ = _decomposition.FactorAnalysis.__doc__

    def summarize(self):
        return '\n'.join([
            "FactorAnalysis (FA)",
            "----------",
            "Number of components:    {n_components}",
            "Log likelihood:          {loglike_}",
            "Noise variance:          {noise_variance_}",
            "Number of iterations: {n_iter_}",
        ]).format(**self.__dict__)
