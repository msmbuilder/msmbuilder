# Author: Muneeb Sultan <msultan>
# Contributors:
# Copyright (c) 2015, Stanford University and the Authors
# All rights reserved.


from sklearn.kernel_approximation import Nystroem

from .base import MultiSequenceDecompositionMixin


__all__ = ['Nystroem']


class Nystroem(MultiSequenceDecompositionMixin, Nystroem):
    __doc__ = Nystroem.__doc__

class LandmarkNystroem(Nystroem):
