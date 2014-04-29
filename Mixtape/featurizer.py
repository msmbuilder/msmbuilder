# Author: Kyle A. Beauchamp <kyleabeauchamp@gmail.com>
# Contributors: Robert McGibbon <rmcgibbo@gmail.com>
# Copyright (c) 2014, Stanford University and the Authors
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are
# met:
#
#   Redistributions of source code must retain the above copyright notice,
#   this list of conditions and the following disclaimer.
#
#   Redistributions in binary form must reproduce the above copyright
#   notice, this list of conditions and the following disclaimer in the
#   documentation and/or other materials provided with the distribution.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS
# IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED
# TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A
# PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
# HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
# SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED
# TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
# PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
# LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
# NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
# SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

#-----------------------------------------------------------------------------
# Imports
#-----------------------------------------------------------------------------
from __future__ import print_function, division, absolute_import

from six.moves import cPickle
import numpy as np
import mdtraj as md

#-----------------------------------------------------------------------------
# Code
#-----------------------------------------------------------------------------


def featurize_all(filenames, featurizer, topology, chunk=1000, stride=1):
    """Load and featurize many trajectory files.

    Parameters
    ----------
    filenames : list of strings
        List of paths to MD trajectory files
    featurizer : Featurizer
        The featurizer to be invoked on each trajectory trajectory as
        it is loaded
    topology : str, Topology, Trajectory
        Topology or path to a topology file, used to load trajectories with
        MDTraj
    chunk : {int, None}
        If chunk is an int, load the trajectories up in chunks using
        md.iterload for better memory efficiency (less trajectory data needs
        to be in memory at once)
    stride : int, default=1
        Only read every stride-th frame.

    Returns
    -------
    data : np.ndarray, shape=(total_length_of_all_trajectories, n_features)
    indices : np.ndarray, shape=(total_length_of_all_trajectories)
    fns : np.ndarray shape=(total_length_of_all_trajectories)
        These three arrays all share the same indexing, such that data[i] is
        the featurized version of indices[i]-th frame in the MD trajectory
        with filename fns[i].
    """
    data = []
    indices = []
    fns = []

    for file in filenames:
        kwargs = {} if file.endswith('.h5') else {'top': topology}
        count = 0
        for t in md.iterload(file, chunk=chunk, stride=stride, **kwargs):
            x = featurizer.featurize(t)
            n_frames = len(x)

            data.append(x)
            indices.append(count + (stride*np.arange(n_frames)))
            fns.extend([file] * n_frames)
            count += (stride*n_frames)
    if len(data) == 0:
        raise ValueError("None!")

    return np.concatenate(data), np.concatenate(indices), np.array(fns)


def load(filename):
    """Load a featurizer from a cPickle file."""
    with open(filename, 'rb') as f:
        featurizer = cPickle.load(f)
    return featurizer


class Featurizer(object):

    """Base class for Featurizer objects."""

    def __init__(self):
        pass

    def featurize(self, traj):
        pass

    def save(self, filename):
        with open(filename, 'wb') as f:
            cPickle.dump(self, f)


class SuperposeFeaturizer(Featurizer):

    """Featurizer based on euclidian atom distances to reference structure."""

    def __init__(self, atom_indices, reference_traj):
        self.atom_indices = atom_indices
        self.reference_traj = reference_traj
        self.n_features = len(self.atom_indices)

    def featurize(self, traj):

        traj.superpose(self.reference_traj, atom_indices=self.atom_indices)
        diff2 = (traj.xyz[:, self.atom_indices] -
                 self.reference_traj.xyz[0, self.atom_indices]) ** 2
        x = np.sqrt(np.sum(diff2, axis=2))

        return x


class AtomPairsFeaturizer(Featurizer):

    """Featurizer based on atom pair distances."""

    def __init__(self, pair_indices, reference_traj, periodic=False, exponent=1.):
        self.pair_indices = pair_indices
        self.reference_traj = reference_traj
        self.n_features = len(self.pair_indices)
        self.periodic = periodic
        self.exponent = exponent 

    def featurize(self, traj):
        d = md.geometry.compute_distances(traj, self.pair_indices, periodic=self.periodic)
        return d ** self.exponent


class DihedralFeaturizer(Featurizer):

    """Featurizer based on dihedral angles"""

    def __init__(self, types, sincos=True):
        if isinstance(types, str):
            types = [types]
        self.types = types
        self.sincos = sincos

        known = {'phi', 'psi', 'omega', 'chi1', 'chi2', 'chi3', 'chi4'}
        if not set(types).issubset(known):
            raise ValueError('angles must be a subset of %s. you supplied %s' % (
                str(known), str(types)))

    def featurize(self, trajectory):
        x = []
        for a in self.types:
            func = getattr(md, 'compute_%s' % a)
            y = func(trajectory)[1]
            if self.sincos:
                x.extend([np.sin(y), np.cos(y)])
            else:
                x.append(y)
        return np.hstack(x)


class ContactFeaturizer(Featurizer):

    """Featurizer based on residue-residue distances"""

    def __init__(self, contacts='all', scheme='closest-heavy', ignore_nonprotein=True):
        self.contacts = contacts
        self.scheme = scheme
        self.ignore_nonprotein = ignore_nonprotein

    def featurize(self, trajectory):
        distances, _ = md.compute_contacts(trajectory, self.contacts, self.scheme, self.ignore_nonprotein)
        return distances


class RawPositionsFeaturizer(Featurizer):

    def __init__(self, n_features):
        self.n_features = n_features

    def featurize(self, traj):
        return traj.xyz.reshape(len(traj), -1)
