# Authors: Matthew Harrigan <matthew.harrigan@outlook.com>
#          Robert McGibbon <rmcgibbo@gmail.com>
#          Brooke Husic <brookehusic@gmail.com>
# Copyright (c) 2016, Stanford University and the Authors
# All rights reserved.

import itertools
import warnings

import mdtraj as md
import numpy as np

from msmbuilder import libdistance
from .base import Featurizer


class RMSDFeaturizer(Featurizer):
    """Featurizer based on RMSD to one or more reference structures.

    This featurizer inputs a trajectory to be analyzed ('traj') and a
    reference trajectory ('ref') and outputs the RMSD of each frame of
    traj with respect to each frame in ref. The output is a numpy array
    with n_rows = traj.n_frames and n_columns = ref.n_frames.

    Parameters
    ----------
    reference_traj : md.Trajectory
        The reference conformations to superpose each frame with respect to
    atom_indices : np.ndarray, shape=(n_atoms,), dtype=int
        The indices of the atoms to superpose and compute the distances with.
        If not specified, all atoms are used.
    trj0
        Deprecated. Please use reference_traj.
    """

    def __init__(self, reference_traj=None, atom_indices=None, trj0=None):
        if trj0 is not None:
            warnings.warn("trj0 is deprecated. Please use reference_traj",
                          DeprecationWarning)
            reference_traj = trj0
        else:
            if reference_traj is None:
                raise ValueError("Please specify a reference trajectory")

        self.atom_indices = atom_indices
        if self.atom_indices is not None:
            self.sliced_reference_traj = \
                reference_traj.atom_slice(self.atom_indices)
        else:
            self.sliced_reference_traj = reference_traj

    def partial_transform(self, traj):
        """Featurize an MD trajectory into a vector space via distance
        after superposition

        Parameters
        ----------
        traj : mdtraj.Trajectory
            A molecular dynamics trajectory to featurize.

        Returns
        -------
        features : np.ndarray, shape=(n_frames, n_ref_frames)
            The RMSD value of each frame of the input trajectory to be
            featurized versus each frame in the reference trajectory. The
            number of features is the number of reference frames.

        See Also
        --------
        transform : simultaneously featurize a collection of MD trajectories
        """
        if self.atom_indices is not None:
            sliced_traj = traj.atom_slice(self.atom_indices)
        else:
            sliced_traj = traj
        result = libdistance.cdist(sliced_traj,
                                   self.sliced_reference_traj,
                                   'rmsd')
        return result


def rotational(x):
    x = list(x)
    for i in range(len(x)):
        yield tuple(x[i:] + x[:i])


class SymmetryRMSDFeaturizer(Featurizer):
    """Calculate RMSD over a symmetry perturbation

    Given a set of permutation groups and a function that orders those
    groups, calculate the RMSD for each permutation and pick the lowest.

    This can be used for homo-multimer proteins. For example, a homologous
    4-domain channel can be featurized by taking the minimum RMSD over
    the four rotations.

    Parameters
    ----------
    reference_traj : mdtraj.Trajectory
        The reference conformations to superpose each frame with respect to
    chains : iterable
        The chain indices to rotate about (using mdtraj's chain indexing)
    atom_groups : list of array_like
        Manually specify permutation groups by giving a list of atom
        indices.
    symmetry : str or callable
        How to permute groups.
         - rotational: start from a given group and loop around
         - permutation: all permutations of the group
        Callables should accept an iterable of group indices.
    """

    symmetry_types = {
        'rotational': rotational,
        'permutation': itertools.permutations
    }

    def __init__(self, reference_traj, chains=None, atom_groups=None,
                 symmetry='permutation'):
        if (chains is None) == (atom_groups is None):
            raise ValueError("Please specify chains or atom_groups, "
                             "but not both.")

        if not callable(symmetry):
            if symmetry not in self.symmetry_types.keys():
                raise ValueError("symmetry must be callable or one of: {}"
                                 .format(list(self.symmetry_types.keys())))
            symmetry = self.symmetry_types[symmetry]

        if chains is not None:
            atom_groups = [[a.index for a in reference_traj.topology.atoms
                            if a.residue.chain.index == chain_i]
                           for chain_i in chains]

        self.reference_traj = reference_traj
        self.symmetry = symmetry
        self.atom_groups = np.asarray(atom_groups, dtype=np.intp)
        self.permutation_indices = np.asarray(
            list(self.symmetry(range(len(self.atom_groups)))), dtype=np.intp)

    def partial_transform(self, traj):
        return libdistance.cdist_symrmsd(traj, self.reference_traj, self.atom_groups, self.permutation_indices)
