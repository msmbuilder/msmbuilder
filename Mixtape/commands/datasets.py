from __future__ import print_function, absolute_import
import os
import sys
import glob

from ..cmdline import NumpydocClassCommand, argument
from ..datasets import AlanineDipeptide


class DatasetCommand(NumpydocClassCommand):
    def start(self):
        print(self.instance)
        self.instance.cache()


class AlanineDipeptideDatasetCommand(DatasetCommand):
    klass = AlanineDipeptide

