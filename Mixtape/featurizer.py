# Author: Kyle A. Beauchamp <kyleabeauchamp@gmail.com>
# Contributors: Robert McGibbon <rmcgibbo@gmail.com>,
#               Matthew Harrigan <matthew.p.harrigan@gmail.com>
# Copyright (c) 2014, Stanford University and the Authors
# All rights reserved.
#
# Mixtape is free software: you can redistribute it and/or modify
# it under the terms of the GNU Lesser General Public License as
# published by the Free Software Foundation, either version 2.1
# of the License, or (at your option) any later version.
#
# This library is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Lesser General Public License for more details.
#
# You should have received a copy of the GNU Lesser General Public
# License along with Mixtape. If not, see <http://www.gnu.org/licenses/>.

#-----------------------------------------------------------------------------
# Imports
#-----------------------------------------------------------------------------
from __future__ import print_function, division, absolute_import

from six.moves import cPickle
import numpy as np
import mdtraj as md
import sklearn.base, sklearn.pipeline
from sklearn.externals.joblib import Parallel, delayed


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
            x = featurizer.partial_transform(t)
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



class Featurizer(sklearn.base.BaseEstimator, sklearn.base.TransformerMixin):
    """Base class for objects that featurize Trajectories.

    Notes
    -----
    At the bare minimum, a featurizer must implement the `partial_transform(traj)`
    member function.  A `transform(traj_list)` for featurizing multiple
    trajectories in batch will be provided.
    """

    def __init__(self):
        pass

    def featurize(self, traj):
        raise NotImplementedError('This API was removed. Use partial_transform instead')

    def partial_transform(self, traj):
        """Featurize an MD trajectory into a vector space.

        Parameters
        ----------
        traj : mdtraj.Trajectory
            A molecular dynamics trajectory to featurize.

        Returns
        -------
        features : np.ndarray, dtype=float, shape=(n_samples, n_features)
            A featurized trajectory is a 2D array of shape
            `(length_of_trajectory x n_features)` where each `features[i]`
            vector is computed by applying the featurization function
            to the `i`th snapshot of the input trajectory.

        See Also
        --------
        transform : simultaneously featurize a collection of MD trajectories
        """
        pass

    def fit(self, traj_list, y=None):
        return self

    def transform(self, traj_list, y=None):
        """Featurize a several trajectories.

        Parameters
        ----------
        traj_list : list(mdtraj.Trajectory)
            Trajectories to be featurized.

        Returns
        -------
        features : list(np.ndarray), length = len(traj_list)
            The featurized trajectories.  features[i] is the featurized
            version of traj_list[i] and has shape
            (n_samples_i, n_features)
        """
        return [self.partial_transform(traj) for traj in traj_list]

    def save(self, filename):
        with open(filename, 'wb') as f:
            cPickle.dump(self, f)


class SuperposeFeaturizer(Featurizer):
    """Featurizer based on euclidian atom distances to reference structure.

    Parameters
    ----------
    atom_indices : np.ndarray, shape=(n_atoms,), dtype=int
        The indices of the atoms to superpose and compute the distances with
    reference_traj : md.Trajectory
        The reference conformation to superpose each frame with respect to
        (only the first frame in reference_traj is used)
    """

    def __init__(self, atom_indices, reference_traj):
        self.atom_indices = atom_indices
        self.reference_traj = reference_traj
        self.n_features = len(self.atom_indices)

    def partial_transform(self, traj):
        """Featurize an MD trajectory into a vector space via distance
        after superposition

        Parameters
        ----------
        traj : mdtraj.Trajectory
            A molecular dynamics trajectory to featurize.

        Returns
        -------
        features : np.ndarray, dtype=float, shape=(n_samples, n_features)
            A featurized trajectory is a 2D array of shape
            `(length_of_trajectory x n_features)` where each `features[i]`
            vector is computed by applying the featurization function
            to the `i`th snapshot of the input trajectory.

        See Also
        --------
        transform : simultaneously featurize a collection of MD trajectories
        """
        traj.superpose(self.reference_traj, atom_indices=self.atom_indices)
        diff2 = (traj.xyz[:, self.atom_indices] -
                 self.reference_traj.xyz[0, self.atom_indices]) ** 2
        x = np.sqrt(np.sum(diff2, axis=2))
        return x


class AtomPairsFeaturizer(Featurizer):

    """Featurizer based on atom pair distances.

    Parameters
    ----------
    pair_indices : np.ndarray, shape=(n_pairs, 2), dtype=int
        Each row gives the indices of two atoms involved in the interaction.
    periodic : bool, default=False
        If `periodic` is True and the trajectory contains unitcell
        information, we will compute distances under the minimum image
        convention.
    exponent : float
        Modify the distances by raising them to this exponent.
    """

    def __init__(self, pair_indices, periodic=False, exponent=1.):
        # TODO: We might want to implement more error checking here. Or during
        # featurize(). E.g. are the pair_indices supplied valid?
        self.pair_indices = pair_indices
        self.n_features = len(self.pair_indices)
        self.periodic = periodic
        self.exponent = exponent

    def partial_transform(self, traj):
        """Featurize an MD trajectory into a vector space via pairwise
        atom-atom distances

        Parameters
        ----------
        traj : mdtraj.Trajectory
            A molecular dynamics trajectory to featurize.

        Returns
        -------
        features : np.ndarray, dtype=float, shape=(n_samples, n_features)
            A featurized trajectory is a 2D array of shape
            `(length_of_trajectory x n_features)` where each `features[i]`
            vector is computed by applying the featurization function
            to the `i`th snapshot of the input trajectory.

        See Also
        --------
        transform : simultaneously featurize a collection of MD trajectories
        """
        d = md.geometry.compute_distances(traj, self.pair_indices, periodic=self.periodic)
        return d ** self.exponent


class DihedralFeaturizer(Featurizer):
    """Featurizer based on dihedral angles.

    Parameters
    ----------
    types : list of strings
        One or more of ['phi', 'psi', 'omega', 'chi1', 'chi2', 'chi3', 'chi4']
    sincos : bool
        Transform to sine and cosine (double the number of featurizers)
    """

    def __init__(self, types, sincos=True):
        if isinstance(types, str):
            types = [types]
        self.types = types
        self.sincos = sincos

        known = {'phi', 'psi', 'omega', 'chi1', 'chi2', 'chi3', 'chi4'}
        if not set(types).issubset(known):
            raise ValueError('angles must be a subset of %s. you supplied %s' % (
                str(known), str(types)))

    def partial_transform(self, traj):
        """Featurize an MD trajectory into a vector space via calculation
        of dihedral (torsion) angles

        Parameters
        ----------
        traj : mdtraj.Trajectory
            A molecular dynamics trajectory to featurize.

        Returns
        -------
        features : np.ndarray, dtype=float, shape=(n_samples, n_features)
            A featurized trajectory is a 2D array of shape
            `(length_of_trajectory x n_features)` where each `features[i]`
            vector is computed by applying the featurization function
            to the `i`th snapshot of the input trajectory.

        See Also
        --------
        transform : simultaneously featurize a collection of MD trajectories
        """
        x = []
        for a in self.types:
            func = getattr(md, 'compute_%s' % a)
            y = func(traj)[1]
            if self.sincos:
                x.extend([np.sin(y), np.cos(y)])
            else:
                x.append(y)
        return np.hstack(x)


class ContactFeaturizer(Featurizer):
    """Featurizer based on residue-residue distances

    Parameters
    ----------
    contacts : np.ndarray or 'all'
        numpy array containing (0-indexed) residues to compute the
        contacts for. (e.g. np.array([[0, 10], [0, 11]]) would compute
        the contact between residue 0 and residue 10 as well as
        the contact between residue 0 and residue 11.) [NOTE: if no
        array is passed then 'all' contacts are calculated. This means
        that the result will contain all contacts between residues
        separated by at least 3 residues.]
    scheme : {'ca', 'closest', 'closest-heavy'}
        scheme to determine the distance between two residues:
            'ca' : distance between two residues is given by the distance
                between their alpha carbons
            'closest' : distance is the closest distance between any
                two atoms in the residues
            'closest-heavy' : distance is the closest distance between
                any two non-hydrogen atoms in the residues
    ignore_nonprotein : bool
        When using `contact==all`, don't compute contacts between
        "residues" which are not protein (i.e. do not contain an alpha
        carbon).
    """

    def __init__(self, contacts='all', scheme='closest-heavy', ignore_nonprotein=True):
        self.contacts = contacts
        self.scheme = scheme
        self.ignore_nonprotein = ignore_nonprotein

    def partial_transform(self, traj):
        """Featurize an MD trajectory into a vector space via of residue-residue
        distances

        Parameters
        ----------
        traj : mdtraj.Trajectory
            A molecular dynamics trajectory to featurize.

        Returns
        -------
        features : np.ndarray, dtype=float, shape=(n_samples, n_features)
            A featurized trajectory is a 2D array of shape
            `(length_of_trajectory x n_features)` where each `features[i]`
            vector is computed by applying the featurization function
            to the `i`th snapshot of the input trajectory.

        See Also
        --------
        transform : simultaneously featurize a collection of MD trajectories
        """
        distances, _ = md.compute_contacts(traj, self.contacts, self.scheme, self.ignore_nonprotein)
        return distances


class GaussianSolventFeaturizer(Featurizer):
    """Featurizer on weighted pairwise distance between solute and solvent.

    We apply a Gaussian kernel to each solute-solvent pairwise distance
    and sum the kernels for each solute atom, resulting in a vector
    of len(solute_indices).

    The values can be physically interpreted as the degree of solvation
    of each solute atom.

    Parameters
    ----------
    solute_indices : np.ndarray, shape=(n_solute,1)
        Indices of solute atoms
    solvent_indices : np.ndarray, shape=(n_solvent, 1)
        Indices of solvent atoms
    sigma : float
        Sets the length scale for the gaussian kernel
    periodic : bool
        Whether to consider a periodic system in distance calculations


    References
    ----------
    ..[1] Gu, Chen, et al. BMC Bioinformatics 14, no. Suppl 2
    (January 21, 2013): S8. doi:10.1186/1471-2105-14-S2-S8.
    """

    def __init__(self, solute_indices, solvent_indices, sigma, periodic=False):
        self.solute_indices = solute_indices[:, 0]
        self.solvent_indices = solvent_indices[:, 0]
        self.sigma = sigma
        self.periodic = periodic
        self.n_features = len(self.solute_indices)

    def partial_transform(self, traj):
        """Featurize an MD trajectory into a vector space via calculation
        of solvent fingerprints

        Parameters
        ----------
        traj : mdtraj.Trajectory
            A molecular dynamics trajectory to featurize.

        Returns
        -------
        features : np.ndarray, dtype=float, shape=(n_samples, n_features)
            A featurized trajectory is a 2D array of shape
            `(length_of_trajectory x n_features)` where each `features[i]`
            vector is computed by applying the featurization function
            to the `i`th snapshot of the input trajectory.

        See Also
        --------
        transform : simultaneously featurize a collection of MD trajectories
        """
        # The result vector
        fingerprints = np.zeros((traj.n_frames, self.n_features))
        atom_pairs = np.zeros((len(self.solvent_indices), 2))
        sigma = self.sigma

        for i, solute_i in enumerate(self.solute_indices):
            # For each solute atom, calculate distance to all solvent
            # molecules
            atom_pairs[:, 0] = solute_i
            atom_pairs[:, 1] = self.solvent_indices

            distances = md.compute_distances(traj, atom_pairs, periodic=True)
            distances = np.exp(-distances / (2 * sigma * sigma))

            # Sum over water atoms for all frames
            fingerprints[:, i] = np.sum(distances, axis=1)

        return fingerprints


class RawPositionsFeaturizer(Featurizer):

    def __init__(self, n_features):
        self.n_features = n_features

    def partial_transform(self, traj):
        """Featurize an MD trajectory into a vector space with the raw
        cartesian coordinates.

        Parameters
        ----------
        traj : mdtraj.Trajectory
            A molecular dynamics trajectory to featurize.

        Returns
        -------
        features : np.ndarray, dtype=float, shape=(n_samples, n_features)
            A featurized trajectory is a 2D array of shape
            `(length_of_trajectory x n_features)` where each `features[i]`
            vector is computed by applying the featurization function
            to the `i`th snapshot of the input trajectory.

        See Also
        --------
        transform : simultaneously featurize a collection of MD trajectories
        """
        value = traj.xyz.reshape(len(traj), -1)
        if value.shape[1] != self.n_features:
            warnings.warn('wrong n_features in RawPositionsFeaturizer')
        return value


class RMSDFeaturizer(Featurizer):
    """Featurizer based on RMSD to a series of reference frames.

    Parameters
    ----------
    trj0 : mdtraj.Trajectory
        Reference trajectory.  trj0.n_frames gives the number of features
        in this Featurizer.
    atom_indices : np.ndarray, default=None
        Which atom indices to use during RMSD calculation.  If None, MDTraj
        should default to all atoms.

    """

    def __init__(self, trj0, atom_indices=None):
        self.n_features = trj0.n_frames
        self.trj0 = trj0
        self.atom_indices = atom_indices

    def partial_transform(self, traj):
        """Featurize an MD trajectory into a vector space by calculating
        the RMSD to each frame in a reference trajectory.

        Parameters
        ----------
        traj : mdtraj.Trajectory
            A molecular dynamics trajectory to featurize.

        Returns
        -------
        features : np.ndarray, dtype=float, shape=(n_samples, n_features)
            A featurized trajectory is a 2D array of shape
            `(length_of_trajectory x n_features)` where each `features[i]`
            vector is computed by applying the featurization function
            to the `i`th snapshot of the input trajectory.

        See Also
        --------
        transform : simultaneously featurize a collection of MD trajectories
        """
        X = np.zeros((traj.n_frames, self.n_features))

        for frame in range(self.n_features):
            X[:, frame] = md.rmsd(traj, self.trj0, atom_indices=self.atom_indices, frame=frame)
        return X


class DRIDFeaturizer(Featurizer):
    """Featurizer based on distribution of reciprocal interatomic
    distances (DRID)

    Parameters
    ----------
    atom_indices : array-like of ints, default=None
        Which atom indices to use during DRID featurization. If None,
        all atoms are used
    """
    def __init__(self, atom_indices=None):
        self.atom_indices = atom_indices

    def partial_transform(self, traj):
        """Featurize an MD trajectory into a vector space using the distribution
        of reciprocal interatomic distance (DRID) method.

        Parameters
        ----------
        traj : mdtraj.Trajectory
            A molecular dynamics trajectory to featurize.

        Returns
        -------
        features : np.ndarray, dtype=float, shape=(n_samples, n_features)
            A featurized trajectory is a 2D array of shape
            `(length_of_trajectory x n_features)` where each `features[i]`
            vector is computed by applying the featurization function
            to the `i`th snapshot of the input trajectory.

        See Also
        --------
        transform : simultaneously featurize a collection of MD trajectories
        """
        return md.geometry.compute_drid(traj, self.atom_indices)


class TrajFeatureUnion(sklearn.pipeline.FeatureUnion):
    """Mixtape version of sklearn.pipeline.FeatureUnion

    Notes
    -----
    Works on lists of trajectories.
    """
    def fit_transform(self, traj_list, y=None, **fit_params):
        """Fit all transformers using `trajectories`, transform the data
        and concatenate results.

        Parameters
        ----------
        traj_list : list (of mdtraj.Trajectory objects)
            Trajectories to featurize
        y : Unused
            Unused

        Returns
        -------
        Y : list (of np.ndarray)
            Y[i] is the featurized version of X[i]
            Y[i] will have shape (n_samples_i, n_features), where
            n_samples_i is the length of trajectory i and n_features
            is the total (concatenated) number of features in the
            concatenated list of featurizers.
        """
        self.fit(traj_list, y, **fit_params)
        return self.transform(traj_list)


    def transform(self, traj_list):
        """Transform traj_list separately by each transformer, concatenate results.

        Parameters
        ----------
        trajectories : list (of mdtraj.Trajectory objects)
            Trajectories to featurize

        Returns
        -------
        Y : list (of np.ndarray)
            Y[i] is the featurized version of X[i]
            Y[i] will have shape (n_samples_i, n_features), where
            n_samples_i is the length of trajectory i and n_features
            is the total (concatenated) number of features in the
            concatenated list of featurizers.

        """
        Xs = Parallel(n_jobs=self.n_jobs)(
            delayed(sklearn.pipeline._transform_one)(trans, name, traj_list, self.transformer_weights)
            for name, trans in self.transformer_list)

        X_i_stacked = [np.hstack([Xs[feature_ind][trj_ind] for feature_ind in range(len(Xs))]) for trj_ind in range(len(Xs[0]))]

        return X_i_stacked
