from __future__ import absolute_import, print_function, division

import sys
import os
import re
import glob
from os.path import join, exists, expanduser
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
  Date:\t\t{{date}}{% if comments %}\nComments:\t\t{{comments}}{% endif %}

{% if previous %}== Derived from ==
{{previous}}
{% endif %}''')

    def __init__(self, path, mode='r', verbose=False):
        self.path = path
        self.mode = mode
        self.verbose = verbose

        if mode not in ('r', 'w'):
            raise ValueError('mode must be one of "r", "w"')
        if mode == 'w':
            if exists(path):
                raise ValueError('File exists: %s' % path)
            os.makedirs(path)
            self._write_provenance()

    def write_derived(self, out_path, sequences, comments='', fmt=None):
        if fmt is None:
            out_dataset = self.__class__(out_path, mode='w', verbose=self.verbose)
        else:
            out_dataset =  dataset(out_path, mode='w', verbose=self.verbose, fmt=fmt)

        for i, x in enumerate(sequences):
            out_dataset[i] = x
        out_dataset._write_provenance(previous=self.provenance,
                                      comments=comments)
        return out_dataset

    def apply(self, fn):
        for key in self.keys():
            yield fn(self.get(key))

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

        filename = join(self.path, self._ITEM_FORMAT % i)
        if self.verbose:
            print('[NumpydirDataset] loading %s' % filename)
        try:
            return np.load(filename, mmap_mode)
        except IOError as e:
            raise IndexError(e)

    def set(self, i, x):
        if 'w' not in self.mode:
            raise IOError('Dataset not opened for writing')
        filename = join(self.path, self._ITEM_FORMAT % i)
        if self.verbose:
            print('[NumpydirDataset] saving %s' % filename)
        return np.save(filename, x)

    def keys(self):
        for fn in sorted(os.listdir(self.path), key=_keynat):
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
  path:\t\t{{path}}
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

        if isinstance(path, list):
            self.glob_matches = [expanduser(fn) for fn in path]
        else:
            self.glob_matches = sorted(glob.glob(expanduser(path)), key=_keynat)

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

    def iterload(self, i, chunk):
        if self.verbose:
            print('[MDTraj dataset] iterloading %s' % self.filename(i))

        if self._topology is None:
            return md.iterload(
                self.filename(i), chunk=chunk, stride=self.stride,
                atom_indices=self.atom_indices)
        else:
            return md.iterload(
                self.filename(i), chunk=chunk, stride=self.stride,
                atom_indices=self.atom_indices, top=self._topology)

    def keys(self):
        return iter(range(len(self.glob_matches)))

    @property
    def provenance(self):
        return self._PROVENANCE_TEMPLATE.render(self.__dict__)


def dataset(path, mode='r', fmt='dir-npy', verbose=False, **kwargs):
    if fmt == 'dir-npy':
        return NumpyDirDataset(path, mode, verbose, **kwargs)
    elif fmt == 'mdtraj':
        return MDTrajDataset(path, mode, verbose, **kwargs)
    else:
        raise NotImplementedError()


def _keynat(string):
    '''A natural sort helper function for sort() and sorted()
    without using regular expression.
    '''
    r = []
    for c in string:
        if c.isdigit():
            if r and isinstance(r[-1], int):
                r[-1] = r[-1] * 10 + int(c)
            else:
                r.append(int(c))
        else:
            r.append(c)
    return r
