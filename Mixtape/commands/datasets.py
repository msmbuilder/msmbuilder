from __future__ import print_function, absolute_import

from ..cmdline import NumpydocClassCommand
from ..datasets import AlanineDipeptide


class DatasetCommand(NumpydocClassCommand):
    def start(self):
        print(self.instance)
        self.instance.cache()


class AlanineDipeptideDatasetCommand(DatasetCommand):
    _concrete = True
    klass = AlanineDipeptide

