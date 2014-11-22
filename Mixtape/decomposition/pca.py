# Author: Matthew Harrigan <matthew.p.harrigan@gmail.com>
# Contributors:
# Copyright (c) 2014, Stanford University and the Authors
# All rights reserved.
#
# Mixtape is free software: you can redistribute it and/or modify
# it under the terms of the GNU Lesser General Public License as
# published by the Free Software Foundation, either version 2.1
# of the License, or (at your option) any later version.
#
# This library is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU Lesser General Public License for more details.
#
# You should have received a copy of the GNU Lesser General Public
# License along with Mixtape. If not, see <http://www.gnu.org/licenses/>.

from __future__ import print_function, division, absolute_import
from sklearn import decomposition
import numpy as np

from .base import MultiSequenceDecompositionMixin

from ..utils import check_iter_of_sequences

__all__ = ['PCA']

class PCA(MultiSequenceDecompositionMixin, decomposition.PCA):
    __doc__ = decomposition.PCA.__doc__

    def summarize(self):
        return """Principal Components Analysis (PCA)
-----------------------------------
Number of components : {n_components}
Fraction explained variance : {expl_var}
Noise variance : {noise_var}
""".format(n_components=self.n_components_, 
           expl_var=self.explained_variance_ratio_,
           noise_var=self.noise_variance_)
