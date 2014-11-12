from __future__ import print_function, absolute_import

from ..cmdline import NumpydocClassCommand
from ..datasets import (AlanineDipeptide, DoubleWell, QuadWell, FsPeptide,
                        MetEnkephalin)


class DatasetCommand(NumpydocClassCommand):
    def start(self):
        self.instance.cache()


class AlanineDipeptideDatasetCommand(DatasetCommand):
    _concrete = True
    klass = AlanineDipeptide


class _NWellDatasetCommand(DatasetCommand):
    def _random_state_type(self, s):
        if s is not None:
            return int(s)
        else:
            return s


class DoubleWellDatasetCommand(_NWellDatasetCommand):
    _concrete = True
    klass = DoubleWell


class QuadWellDatasetCommand(_NWellDatasetCommand):
    _concrete = True
    klass = QuadWell


class FsPeptideDatasetCommand(DatasetCommand):
    _concrete = True
    klass = FsPeptide


class MetEnkephalinDatasetCommand(DatasetCommand):
    _concrete = True
    klass = MetEnkephalin
