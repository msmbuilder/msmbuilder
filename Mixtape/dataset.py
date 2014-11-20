from __future__ import absolute_import, print_function, division

import sys
import os
from os.path import join, exists
import re
import socket
import getpass
import itertools
import numpy as np
from datetime import datetime
from collections import Sequence

from . import version

__all__ = ['dataset', 'NumpyDirDataset']


class NumpyDirDataset(Sequence):
    """Mixtape dataset container

    Parameters
    ----------
    path : str
    mode : {'r', 'w'}

    Examples
    --------
    for X in Dataset('path/to/dataset'):
        print X
    """

    ITEM_FORMAT = '%08d.npy'
    ITEM_RE = re.compile('\d{8}.npy')
    PROVENANCE_FILE = 'PROVENANCE.txt'
    PROVENANCE_TEMPLATE = \
'''Mixtape:\t{version}
Command:\t{cmdline}
Path:\t\t{path}
Username:\t{user}
Hostname:\t{hostname}
Date:\t\t{date}

{previous}'''

    def __init__(self, path, mode='r', fmt='dir-npy'):
        self.path = path
        self.mode = mode
        self.fmt = fmt

        if mode not in ('r', 'w'):
            raise ValueError('mode must be one of "r", "w"')
        if mode == 'w':
            if exists(path):
                raise ValueError('File exists: %s' % path)
            os.makedirs(path)
            self._write_header()

    def save_transformed(self, out_path, sequences):
        out_dataset = self.__class__(out_path, 'w')
        for i, x in enumerate(sequences):
            out_dataset[i] = x
        out_dataset._write_header(previous=self.provenance())
        return out_dataset

    def provenance(self):
        with open(join(self.path, self.PROVENANCE_FILE), 'r') as f:
            return f.read()

    def _write_header(self, previous=''):
        with open(join(self.path, self.PROVENANCE_FILE), 'w') as f:
            f.write(self.PROVENANCE_TEMPLATE.format(
                version=version.full_version,
                cmdline=' '.join(sys.argv),
                user=getpass.getuser(),
                hostname=socket.gethostname(),
                previous=previous,
                path=self.path,
                date=datetime.now().strftime("%B %d, %Y %l:%M %p")))

    def get(self, i, mmap=False):
        if isinstance(i, slice):
            items = []
            for ii in itertools.islice(*i.indices(len(self))):
                items.append(self.get(ii))
            return items

        mmap_mode = 'r' if mmap else None
        try:
            return np.load(join(self.path, self.ITEM_FORMAT % i), mmap_mode)
        except IOError as e:
            raise IndexError(e)

    def set(self, i, x):
        if 'w' not in self.mode:
            raise IOError('Dataset not opened for writing')
        return np.save(join(self.path, self.ITEM_FORMAT % i), x)

    def __len__(self):
        # this is probably slow
        return sum(1 for e in os.listdir(self.path) if self.ITEM_RE.match(e))

    def __getitem__(self, i):
        return self.get(i)

    def __setitem__(self, i, x):
        return self.set(i, x)

    def append(self, x):
        i = len(self)
        self.set(i, x)

    def extend(self, sequences):
        i = len(self)
        for x in sequences:
            self.set(i, x)
            i += 1


def dataset(path, mode='r', fmt='dir-npy'):
    if fmt == 'dir-npy':
        return NumpyDirDataset(path, mode)
    else:
        raise NotImplementedError()
