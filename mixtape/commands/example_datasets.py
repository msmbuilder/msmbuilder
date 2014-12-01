from __future__ import print_function, absolute_import

from ..cmdline import NumpydocClassCommand
from ..example_datasets import (AlanineDipeptide, DoubleWell, QuadWell, FsPeptide,
                                MetEnkephalin)


class DatasetCommand(NumpydocClassCommand):
    _group = 'Dataset'
    def start(self):
        self.instance.cache()


class AlanineDipeptideDatasetCommand(DatasetCommand):
    _concrete = True
    klass = AlanineDipeptide
    description = 'Download example alanine dipeptide dataset.'


class _NWellDatasetCommand(DatasetCommand):
    def _random_state_type(self, s):
        if s is not None:
            return int(s)
        else:
            return s


class DoubleWellDatasetCommand(_NWellDatasetCommand):
    _concrete = True
    klass = DoubleWell
    description = 'Generate example double well potential dataset.'


class QuadWellDatasetCommand(_NWellDatasetCommand):
    _concrete = True
    klass = QuadWell
    description = 'Generate example quad-well potential dataset.'


class FsPeptideDatasetCommand(DatasetCommand):
    _concrete = True
    klass = FsPeptide
    description = 'Download example Fs-peptide dataset.'


class MetEnkephalinDatasetCommand(DatasetCommand):
    _concrete = True
    klass = MetEnkephalin
    description = 'Download example Met-Enkephalin dataset.'
