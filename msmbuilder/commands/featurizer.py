from __future__ import print_function, absolute_import
import os
import warnings

import numpy as np
import mdtraj as md

from ..utils.progressbar import ProgressBar, Percentage, Bar, ETA
from ..utils import verbosedump
from ..cmdline import NumpydocClassCommand, argument, exttype, stripquotestype
from ..dataset import dataset, MDTrajDataset
from ..featurizer import (AtomPairsFeaturizer, SuperposeFeaturizer,
                          DRIDFeaturizer, DihedralFeaturizer,
                          ContactFeaturizer, GaussianSolventFeaturizer,
                          KappaAngleFeaturizer, AlphaAngleFeaturizer,
                          RMSDFeaturizer, BinaryContactFeaturizer,
                          LogisticContactFeaturizer, VonMisesFeaturizer,
                          FunctionFeaturizer, RawPositionsFeaturizer,
                          SASAFeaturizer)


class FeaturizerCommand(NumpydocClassCommand):
    _group = '1-Featurizer'
    trjs = argument(
        '--trjs', help='Glob pattern for trajectories',
        default='', required=True, type=stripquotestype)
    top = argument(
        '--top', help='Path to topology file matching the trajectories', default='')
    chunk = argument(
        '--chunk',
        help='''Chunk size for loading trajectories using mdtraj.iterload''',
        default=10000, type=int)
    out = argument(
        '-o', '--out', help='''Path to save featurizer instance using
        the pickle protocol''',
        default='', type=exttype('.pkl'))
    transformed = argument(
        '--transformed',
        help="Output path for transformed data",
        type=exttype('/'), required=True)
    stride = argument(
        '--stride', default=1, type=int,
        help='Load only every stride-th frame')

    def start(self):
        if os.path.exists(self.transformed):
            self.error('File exists: %s' % self.transformed)
        if os.path.exists(self.out):
            self.error('File exists: %s' % self.out)

        print(self.instance)
        if self.top.strip() == "":
            top = None
        else:
            top = os.path.expanduser(self.top)
            err = "Couldn't find topology file '{}'".format(top)
            assert os.path.exists(top), err

        input_dataset = MDTrajDataset(self.trjs, topology=top, stride=self.stride, verbose=False)
        out_dataset = input_dataset.create_derived(self.transformed, fmt='dir-npy')

        pbar = ProgressBar(widgets=[Percentage(), Bar(), ETA()],
                           maxval=len(input_dataset)).start()
        for key in pbar(input_dataset.keys()):
            trajectory = []
            for i, chunk in enumerate(input_dataset.iterload(key, chunk=self.chunk)):
                trajectory.append(self.instance.partial_transform(chunk))
            out_dataset[key] = np.concatenate(trajectory)
            out_dataset.close()

        print("\nSaving transformed dataset to '%s'" % self.transformed)
        print("To load this dataset interactive inside an IPython")
        print("shell or notebook, run\n")
        print("  $ ipython")
        print("  >>> from msmbuilder.dataset import dataset")
        print("  >>> ds = dataset('%s')\n" % self.transformed)

        if self.out != '':
            verbosedump(self.instance, self.out)
            print("To load this %s object interactively inside an IPython\n"
                  "shell or notebook, run: \n" % self.klass.__name__)
            print("  $ ipython")
            print("  >>> from msmbuilder.utils import load")
            print("  >>> model = load('%s')\n" % self.out)


class DihedralFeaturizerCommand(FeaturizerCommand):
    _concrete = True
    klass = DihedralFeaturizer
    example = '''
    $ msmb DihedralFeaturizer --trjs './trajectories/*.h5' \\
        --transformed dihedrals-withchi --types phi psi chi1
    '''

class KappaAngleFeaturizerCommand(FeaturizerCommand):
    _concrete = True
    klass = KappaAngleFeaturizer


class AlphaAngleFeaturizerCommand(FeaturizerCommand):
    _concrete = True
    klass = AlphaAngleFeaturizer


class AtomPairsFeaturizerCommand(FeaturizerCommand):
    klass = AtomPairsFeaturizer
    _concrete = True

    def _pair_indices_type(self, fn):
        if fn is None:
            return None
        return np.loadtxt(fn, dtype=int, ndmin=2)


class RMSDFeaturizerCommand(FeaturizerCommand):
    klass = RMSDFeaturizer
    _concrete = True

    def _reference_traj_type(self, fn):
        if self.top.strip() == "":
            top = None
        else:
            top = os.path.expanduser(self.top)
            err = ("Couldn't find topology file '{}' "
                   "when loading reference trajectory".format(top))
            assert os.path.exists(top), err
        return md.load(fn, top=top)

    def _atom_indices_type(self, fn):
        if fn is None:
            return None
        return np.loadtxt(fn, dtype=int, ndmin=1)


class SuperposeFeaturizerCommand(FeaturizerCommand):
    klass = SuperposeFeaturizer
    _concrete = True

    def _reference_traj_type(self, fn):
        if self.top.strip() == "":
            top = None
        else:
            top = os.path.expanduser(self.top)
            err = ("Couldn't find topology file '{}' "
                   "when loading reference trajectory".format(top))
            assert os.path.exists(top), err
        return md.load(fn, top=top)

    def _atom_indices_type(self, fn):
        if fn is None:
            return None
        return np.loadtxt(fn, dtype=int, ndmin=1)


class DRIDFeaturizerCommand(FeaturizerCommand):
    klass = DRIDFeaturizer
    _concrete = True

    def _atom_indices_type(self, fn):
        if fn is None:
            return None
        return np.loadtxt(fn, dtype=int, ndmin=1)


class ContactFeaturizerCommand(FeaturizerCommand):
    _concrete = True
    klass = ContactFeaturizer

    def _contacts_type(self, val):
        if val == 'all':
            return val
        else:
            return np.loadtxt(val, dtype=int, ndmin=2)


class BinaryContactFeaturizerCommand(FeaturizerCommand):
    _concrete = True
    klass = BinaryContactFeaturizer

    def _contacts_type(self, val):
        if val == 'all':
            return val
        else:
            return np.loadtxt(val, dtype=int, ndmin=2)


class LogisticContactFeaturizerCommand(FeaturizerCommand):
    _concrete = True
    klass = LogisticContactFeaturizer

    def _contacts_type(self, val):
        if val == 'all':
            return val
        else:
            return np.loadtxt(val, dtype=int, ndmin=2)


class GaussianSolventFeaturizerCommand(FeaturizerCommand):
    _concrete = True
    klass = GaussianSolventFeaturizer

    def _solvent_indices_type(self, fn):
        return np.loadtxt(fn, dtype=int, ndmin=1)

    def _solute_indices_type(self, fn):
        return np.loadtxt(fn, dtype=int, ndmin=1)


class VonMisesFeaturizerCommand(FeaturizerCommand):
    _concrete = True
    klass = VonMisesFeaturizer


class RawPositionsFeaturizerCommand(FeaturizerCommand):
    klass = RawPositionsFeaturizer
    _concrete = True

    def _reference_traj_type(self, fn):
        if self.top.strip() == "":
            top = None
        else:
            top = os.path.expanduser(self.top)
            err = ("Couldn't find topology file '{}' "
                   "when loading reference trajectory".format(top))
            assert os.path.exists(top), err
        return md.load(fn, top=top)

    def _atom_indices_type(self, fn):
        if fn is None:
            return None
        return np.loadtxt(fn, dtype=int, ndmin=1)


class SASAFeaturizerCommand(FeaturizerCommand):
    _concrete = True
    klass = SASAFeaturizer
