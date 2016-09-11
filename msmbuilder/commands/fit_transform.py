# Author: Robert McGibbon <rmcgibbo@gmail.com>
# Contributors: Brooke Husic <brookehusic@gmail.com>
# Copyright (c) 2014, Stanford University
# All rights reserved.

# -----------------------------------------------------------------------------
# Imports
# -----------------------------------------------------------------------------

from __future__ import print_function, absolute_import

import os
import numpy as np

from ..utils.progressbar import ProgressBar, Percentage, Bar, ETA
from ..dataset import dataset, _guess_format
from ..utils import verbosedump

try:
    from ..preprocessing import StandardScaler, RobustScaler
except:
    StandardScaler = RobustScaler = None

from ..preprocessing import Butterworth, DoubleEWMA
from ..decomposition import (tICA, SparseTICA, FastICA, FactorAnalysis,
                             KernelTICA, PCA, SparsePCA, MiniBatchSparsePCA)
from ..cluster import (KMeans, KCenters, KMedoids, MiniBatchKMedoids,
                       MiniBatchKMeans, RegularSpatial, LandmarkAgglomerative,
                       GMM, MeanShift, NDGrid, SpectralClustering,
                       AffinityPropagation, APM, AgglomerativeClustering)

from ..cmdline import (NumpydocClassCommand, argument_group,
                       exttype, stripquotestype)


class FitTransformCommand(NumpydocClassCommand):
        # What format to use for saving transformed dataset
    _transformed_fmt = 'hdf5'

    g1 = argument_group('input')
    inp = g1.add_argument(
        '-i', '--inp', help='''Path to input dataset. Each dataset is a
        collection of sequences, which may either be an array or Trajectory,
        depending on the model.''', required=True)
    g2 = argument_group('output', description='use one or both of the following')
    out = g2.add_argument(
        '-o', '--out', help='''Path to save fit model instance. This is the
        model, after parameterization with fit(), saved using the pickle
        protocol''',
        default='', type=exttype('.pkl'))
    transformed = g2.add_argument(
        '-t', '--transformed', help='''Path to output transformed dataset. This
        will be a collection of arrays, as transfomed by the model''',
        default='', type=exttype('.h5'))

    def load_dataset(self):
        return dataset(self.inp, mode='r', verbose=False)

    def start(self):
        if self.out is '' and self.transformed is '':
            self.error('One of --out or --model should be specified')
        if self.transformed is not '' and os.path.exists(self.transformed):
            self.error('File exists: %s' % self.transformed)

        inp_ds = self.load_dataset()
        print(self.instance)

        print("Fitting model...")
        self.instance.fit(inp_ds)

        print("*********\n*RESULTS*\n*********")
        print(self.instance.summarize())
        print('-' * 80)

        if self.transformed is not '':
            out_ds = inp_ds.create_derived(self.transformed, fmt=self._transformed_fmt)
            pbar = ProgressBar(
                widgets=['Transforming ', Percentage(), Bar(), ETA()],
                maxval=len(inp_ds)).start()

            for key in pbar(inp_ds.keys()):
                in_seq = inp_ds.get(key)
                out_ds[key] = self.instance.partial_transform(in_seq)
            out_ds.close()

            print("\nSaving transformed dataset to '%s'" % self.transformed)
            print("To load this dataset interactive inside an IPython")
            print("shell or notebook, run\n")
            print("  $ ipython")
            print("  >>> from msmbuilder.dataset import dataset")
            print("  >>> ds = dataset('%s')\n" % self.transformed)

        if self.out is not '':
            verbosedump(self.instance, self.out)
            print("To load this %s object interactively inside an IPython\n"
                  "shell or notebook, run: \n" % self.klass.__name__)
            print("  $ ipython")
            print("  >>> from msmbuilder.utils import load")
            print("  >>> model = load('%s')\n" % self.out)

        inp_ds.close()



class TrajectoryClusterCommand(FitTransformCommand):
    # A fit-transform for clustering that can either accept a collection
    # of arrays _or_ a collection of Trajectories to fit()
    FitTransformCommand.g1.replace_argument(
        '-i', '--inp', required=True, help='''Input dataset. This can either be a
        glob pattern for trajectories (if using metric='rmsd'), or a dataset
        containing a collection of .npy files, for other metrics.''',
        type=stripquotestype)
    md = argument_group('mdtraj input', description='If using metric="rmsd", '
        'additional options required for loading trajectories')
    md.add_argument('--stride', default=1, help='Load only every stride-th frame', type=int)
    md.add_argument('--top', default='', help='Path to topology file matching the trajectories')
    md.add_argument('--atom_indices', help='''Path to an index file
        containing the zero-based indices of the atoms to use for
        RMSD''')

    def _random_state_type(self, state):
        if state is None:
            return None
        return int(state)

    def _atom_indices_type(self, fn):
        if fn is None:
            return None
        arr = np.loadtxt(fn, dtype=int, ndmin=1)
        if not arr.ndim == 1:
            raise ValueError('%s has incorrect dimension' % self.fn)
        return fn

    def load_dataset(self):
        if _guess_format(self.inp) == 'mdtraj':
            if self.instance.metric != 'rmsd':
                self.error("Direct use of a trajectory dataset by this model "
                           "requires metric='rmsd'")

        if hasattr(self.instance, 'metric') and self.instance.metric == 'rmsd':
            # use mdtraj format dataset for metric='rmsd'.
            # THIS IS KIND OF A HACK
            return dataset(self.inp, fmt='mdtraj', topology=self.top,
                           stride=self.stride, verbose=False)
        return dataset(self.inp, mode='r', verbose=False)


class ButterworthCommand(FitTransformCommand):
    klass = Butterworth
    _concrete = True
    _group = '1-Preprocessing'
    _transformed_fmt = 'hdf5'


class DoubleEWMACommand(FitTransformCommand):
    klass = DoubleEWMA
    _concrete = True
    _group = '1-Preprocessing'
    _transformed_fmt = 'hdf5'

if StandardScaler is not None:
    class StandardScalerCommand(FitTransformCommand):
        klass = StandardScaler
        _concrete = True
        _group = '1-Preprocessing'
        _transformed_fmt = 'hdf5'

if RobustScaler is not None:
    class RobustScalerCommand(FitTransformCommand):
        klass = RobustScaler
        _concrete = True
        _group = '1-Preprocessing'
        _transformed_fmt = 'hdf5'


class tICACommand(FitTransformCommand):
    klass = tICA
    _concrete = True
    _group = '3-Decomposition'
    _transformed_fmt = 'hdf5'


class SparseTICACommand(FitTransformCommand):
    klass = SparseTICA
    _concrete = True
    _group = '3-Decomposition'
    _transformed_fmt = 'hdf5'


class FastICACommand(FitTransformCommand):
    klass = FastICA
    _concrete = True
    _group = '3-Decomposition'
    _transformed_fmt = 'hdf5'


class FactorAnalysisCommand(FitTransformCommand):
    klass = FactorAnalysis
    _concrete = True
    _group = '3-Decomposition'
    _transformed_fmt = 'hdf5'


class KernelTICACommand(FitTransformCommand):
    klass = KernelTICA
    _concrete = True
    _group = '3-Decomposition'
    _transformed_fmt = 'hdf5'


class PCACommand(FitTransformCommand):
    klass = PCA
    _concrete = True
    _group = '3-Decomposition'
    _transformed_fmt = 'hdf5'


class SparsePCACommand(FitTransformCommand):
    klass = SparsePCA
    _concrete = True
    _group = '3-Decomposition'
    _transformed_fmt = 'hdf5'


class MiniBatchSparsePCACommand(FitTransformCommand):
    klass = MiniBatchSparsePCA
    _concrete = True
    _group = '3-Decomposition'
    _transformed_fmt = 'hdf5'


class KMeansCommand(FitTransformCommand):
    klass = KMeans
    _concrete = True
    _group = '2-Clustering'
    _transformed_fmt = 'hdf5'

    def _random_state_type(self, state):
        if state is None:
            return None
        return int(state)


class MiniBatchKMeansCommand(KMeansCommand):
    klass = MiniBatchKMeans
    _concrete = True
    _group = '2-Clustering'
    _transformed_fmt = 'hdf5'


class KCentersCommand(TrajectoryClusterCommand):
    klass = KCenters
    _concrete = True
    _group = '2-Clustering'
    _transformed_fmt = 'hdf5'


class KMedoidsCommand(TrajectoryClusterCommand):
    klass = KMedoids
    _concrete = True
    _group = '2-Clustering'
    _transformed_fmt = 'hdf5'


class MiniBatchKMedoidsCommand(TrajectoryClusterCommand):
    klass = MiniBatchKMedoids
    _concrete = True
    _group = '2-Clustering'
    _transformed_fmt = 'hdf5'


class RegularSpatialCommand(TrajectoryClusterCommand):
    klass = RegularSpatial
    _concrete = True
    _group = '2-Clustering'
    _transformed_fmt = 'hdf5'


class LandmarkAgglomerativeCommand(TrajectoryClusterCommand):
    klass = LandmarkAgglomerative
    _concrete = True
    _group = '2-Clustering'
    _transformed_fmt = 'hdf5'


class GMMCommand(TrajectoryClusterCommand):
    klass = GMM
    _concrete = True
    _group = '2-Clustering'
    _transformed_fmt = 'hdf5'


class MeanShiftCommand(TrajectoryClusterCommand):
    klass = MeanShift
    _concrete = True
    _group = '2-Clustering'
    _transformed_fmt = 'hdf5'


class NDGridCommand(TrajectoryClusterCommand):
    klass = NDGrid
    _concrete = True
    _group = '2-Clustering'
    _transformed_fmt = 'hdf5'


class SpectralClusteringCommand(TrajectoryClusterCommand):
    klass = SpectralClustering
    _concrete = True
    _group = '2-Clustering'
    _transformed_fmt = 'hdf5'


class AffinityPropagationCommand(TrajectoryClusterCommand):
    klass = AffinityPropagation
    _concrete = True
    _group = '2-Clustering'
    _transformed_fmt = 'hdf5'


class APMCommand(TrajectoryClusterCommand):
    klass = APM
    _concrete = True
    _group = '2-Clustering'
    _transformed_fmt = 'hdf5'


class AgglomerativeClusteringCommand(TrajectoryClusterCommand):
    klass = AgglomerativeClustering
    _concrete = True
    _group = '2-Clustering'
    _transformed_fmt = 'hdf5'
