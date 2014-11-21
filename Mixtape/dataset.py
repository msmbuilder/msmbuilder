from __future__ import absolute_import, print_function, division

import sys
import os
import re
import glob
from os.path import join, exists
import socket
import getpass
import itertools
from datetime import datetime
from collections import Sequence

import mdtraj as md
from mdtraj.core.trajectory import _parse_topology
import numpy as np
from jinja2 import Template
from . import version

__all__ = ['dataset', 'NumpyDirDataset']


class _BaseDataset(Sequence):
    _PROVENANCE_TEMPLATE = Template(
        '''Mixtape Dataset:
  Mixtape:\t{{version}}
  Command:\t{{cmdline}}
  Path:\t\t{{path}}
  Username:\t{{user}}
  Hostname:\t{{hostname}}
  Date:\t\t{{date}}
  {% if comments %}Comments:\t\t{{comments}}{% endif %}

{% if previous %}
== Derived from ==
  {{previous}}{% endif %}''')

    def __init__(self, path, mode='r'):
        self.path = path
        self.mode = mode

        if mode not in ('r', 'w'):
            raise ValueError('mode must be one of "r", "w"')
        if mode == 'w':
            if exists(path):
                raise ValueError('File exists: %s' % path)
            os.makedirs(path)
            self._write_provenance()

    def write_derived(self, out_path, sequences, comments=''):
        out_dataset = self.__class__(out_path, 'w')
        for i, x in enumerate(sequences):
            out_dataset[i] = x
        out_dataset._write_provenance(previous=self.provenance,
                                      comments=comments)
        return out_dataset

    def _build_provenance(self, previous=None, comments=''):
        return self._PROVENANCE_TEMPLATE.render(
            version=version.full_version,
            cmdline=' '.join(sys.argv),
            user=getpass.getuser(),
            hostname=socket.gethostname(),
            previous=previous,
            path=self.path,
            comments=comments,
            date=datetime.now().strftime("%B %d, %Y %l:%M %p"))

    @property
    def provenance(self):
        raise NotImplementedError('implemented in subclass')

    def __len__(self):
        return sum(1 for xx in self.keys())

    def __getitem__(self, i):
        return self.get(i)

    def __setitem__(self, i, x):
        return self.set(i, x)

    def __iter__(self):
        for key in self.keys():
            yield self.get(key)

    def keys(self):
        raise NotImplementedError('implemeneted in subclass')

    def get(self, i):
        raise NotImplementedError('implemeneted in subclass')

    def set(self, i, x):
        raise NotImplementedError('implemeneted in subclass')


class NumpyDirDataset(_BaseDataset):
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

    _ITEM_FORMAT = '%08d.npy'
    _ITEM_RE = re.compile('(\d{8}).npy')
    _PROVENANCE_FILE = 'PROVENANCE.txt'

    def get(self, i, mmap=False):
        if isinstance(i, slice):
            items = []
            for ii in itertools.islice(*i.indices(len(self))):
                items.append(self.get(ii))
            return items

        mmap_mode = 'r' if mmap else None
        try:
            return np.load(join(self.path, self._ITEM_FORMAT % i), mmap_mode)
        except IOError as e:
            raise IndexError(e)

    def set(self, i, x):
        if 'w' not in self.mode:
            raise IOError('Dataset not opened for writing')
        return np.save(join(self.path, self._ITEM_FORMAT % i), x)

    def keys(self):
        for fn in os.listdir(self.path):
            match = self._ITEM_RE.match(fn)
            if match:
                yield int(match.group(1))

    @property
    def provenance(self):
        try:
            with open(join(self.path, self._PROVENANCE_FILE), 'r') as f:
                return f.read()
        except IOError:
            return 'No available provenance'

    def _write_provenance(self, previous=None, comments=''):
        with open(join(self.path, self._PROVENANCE_FILE), 'w') as f:
            f.write(self._build_provenance(previous=previous,
                                           comments=comments))


class MDTrajDataset(_BaseDataset):
    _PROVENANCE_TEMPLATE = Template(
        '''MDTraj dataset:
  path:\t{{path}}
  topology:\t{{topology}}
  stride:\t{{stride}}
  atom_indices\t{{atom_indices}}
''')

    def __init__(self, path, mode='r', topology=None, stride=1,
                 atom_indices=None, verbose=False):
        if mode != 'r':
            raise ValueError('mode must be "r"')
        self.path = path
        self.topology = topology
        self.stride = stride
        self.atom_indices = atom_indices
        self.verbose = verbose
        self.glob_matches = glob.glob(path)
        if topology is None:
            self._topology = None
        else:
            self._topology = _parse_topology(topology)

    def get(self, i):
        if self.verbose:
            print('[MDTraj dataset] loading %s' % self.filename(i))

        if self._topology is None:
            return md.load(self.filename(i), stride=self.stride,
                           atom_indices=self.atom_indices)
        else:
            return md.load(self.filename(i), stride=self.stride,
                           atom_indices=self.atom_indices, top=self._topology)

    def filename(self, i):
        return self.glob_matches[i]

    def keys(self):
        return list(range(len(self.glob_matches)))

    @property
    def provenance(self):
        return self._PROVENANCE_TEMPLATE.render(self.__dict__)


def dataset(path, mode='r', fmt='dir-npy', **kwargs):
    if fmt == 'dir-npy':
        return NumpyDirDataset(path, mode, **kwargs)
    elif fmt == 'mdtraj':
        return MDTrajDataset(path, mode, **kwargs)
    else:
        raise NotImplementedError()
