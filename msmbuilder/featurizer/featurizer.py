# Author: Kyle A. Beauchamp <kyleabeauchamp@gmail.com>
# Contributors: Robert McGibbon <rmcgibbo@gmail.com>,
#               Matthew Harrigan <matthew.p.harrigan@gmail.com>
#               Brooke Husic <brookehusic@gmail.com>,
#               Muneeb Sultan <msultan@stanford.edu>
# Copyright (c) 2016, Stanford University and the Authors
# All rights reserved.

from __future__ import print_function, division, absolute_import

import warnings

import mdtraj as md
import numpy as np
import sklearn.pipeline
from scipy.stats import vonmises as vm
from msmbuilder import libdistance
import itertools
from sklearn.base import TransformerMixin
from sklearn.externals.joblib import Parallel, delayed

from msmbuilder import libdistance
from ..base import BaseEstimator

def zippy_maker(aind_tuples, top):
    resseqs = []
    resids = []
    resnames = []
    for ainds in aind_tuples:
        resid = set(top.atom(ai).residue.index for ai in ainds)
        resids += [list(resid)]
        reseq = set(top.atom(ai).residue.resSeq for ai in ainds)
        resseqs += [list(reseq)]
        resname = set(top.atom(ai).residue.name for ai in ainds)
        resnames += [list(resname)]

    return zip(aind_tuples, resseqs, resids, resnames)

def dict_maker(zippy):
    feature_descs = []
    for featurizer, featuregroup, other_info, feature_info in zippy:
        ainds, resseq, resid, resname = feature_info
        feature_descs += [dict(
            resnames=resname,
            atominds=ainds,
            resseqs=resseq,
            resids=resid,
            featurizer=featurizer,
            featuregroup="{}".format(featuregroup),
            otherinfo ="{}".format(other_info)
        )]
    return feature_descs

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
            indices.append(count + (stride * np.arange(n_frames)))
            fns.extend([file] * n_frames)
            count += (stride * n_frames)
    if len(data) == 0:
        raise ValueError("None!")

    return np.concatenate(data), np.concatenate(indices), np.array(fns)

class Featurizer(BaseEstimator, TransformerMixin):
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

    def describe_features(self, traj):
        """Generic method for describing features.

        Parameters
        ----------
        traj : mdtraj.Trajectory
            Trajectory to use

        Returns
        -------
        feature_descs : list of dict
            Dictionary describing each feature with the following information
            about the atoms participating in each feature
                - resnames: unique names of residues
                - atominds: the four atom indicies
                - resseqs: unique residue sequence ids (not necessarily
                  0-indexed)
                - resids: unique residue ids (0-indexed)
                - featurizer: Featurizer name
                - featuregroup: Other information

        Notes
        -------
        Method resorts to returning N/A for everything if describe_features in not
        implemented in the sub_class
        """
        n_f = self.partial_transform(traj).shape[1]
        zippy=zip(itertools.repeat("N/A", n_f),
                  itertools.repeat("N/A", n_f),
                  itertools.repeat("N/A", n_f),
                  itertools.repeat(("N/A","N/A","N/A","N/A"), n_f))

        return dict_maker(zippy)

class SuperposeFeaturizer(Featurizer):
    """Featurizer based on euclidian atom distances to reference structure.

    This featurizer transforms a dataset containing MD trajectories into
    a vector dataset by representing each frame in each of the MD trajectories
    by a vector containing the distances from a specified set of atoms to
    the 'reference position' of those atoms, in ``reference_traj``.

    Parameters
    ----------
    atom_indices : np.ndarray, shape=(n_atoms,), dtype=int
        The indices of the atoms to superpose and compute the distances with
    reference_traj : md.Trajectory
        The reference conformation to superpose each frame with respect to
        (only the first frame in reference_traj is used)
    superpose_atom_indices : np.ndarray, shape=(n_atoms,), dtype=int
        If not None, these atom_indices are used for the superposition
    """

    def __init__(self, atom_indices, reference_traj, superpose_atom_indices=None):
        self.atom_indices = atom_indices
        if superpose_atom_indices is None:
            self.superpose_atom_indices = atom_indices
        else:
            self.superpose_atom_indices = superpose_atom_indices
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
        traj.superpose(self.reference_traj,
                       atom_indices=self.superpose_atom_indices)
        diff2 = (traj.xyz[:, self.atom_indices] -
                 self.reference_traj.xyz[0, self.atom_indices]) ** 2
        x = np.sqrt(np.sum(diff2, axis=2))
        return x


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
            self.sliced_reference_traj = reference_traj.atom_slice(self.atom_indices)
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
        result = libdistance.cdist(
            sliced_traj, self.sliced_reference_traj, 'rmsd'
        )
        return result


class AtomPairsFeaturizer(Featurizer):
    """Featurizer based on distances between specified pairs of atoms.

    This featurizer transforms a dataset containing MD trajectories into
    a vector dataset by representing each frame in each of the MD trajectories
    by a vector of the distances between the specified pairs of atoms.

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
        self.atom_indices = pair_indices
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
        d = md.geometry.compute_distances(traj, self.pair_indices,
                                          periodic=self.periodic)
        return d ** self.exponent


class FunctionFeaturizer(Featurizer):
    """Featurizer based on arbitrary functions.

    This featurizer transforms a dataset containing MD trajectories into
    a vector dataset by representing each frame in each of the MD trajectories
    by a vector the output of the function.

    Parameters
    ----------
    function : function
        Instantiation of the function. The function should accept
        a mdtraj.Trajectory object as the first argument.
    func_args : dictionary
        A dictionary of key word arguments(keys) and their values to
        pass to the function. These should NOT include the trajectory
        object which is passed in as the first argument.

    Notes
    ----------
    This Featurizer assumes that the function takes in the trajectory object
    as the first argument.

    Examples
    --------
    >>> function = compute_dihedrals
    >>> f = FunctionFeaturizer(function, func_args={indices: [[0,1,2,3]]})
    >>> results = f.transform(dataset)
    """

    def __init__(self, function, func_args={}):
        if callable(function):
            self.function = function
            self.func_args = func_args
        else:
            raise ValueError("Sorry but we "
                             "couldn't use the "
                             "provided function "
                             "because it is not "
                             "callable")

    def partial_transform(self, traj):
        """Featurize an MD trajectory using the provided function.

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

        Notes
        -----
        This method assumes that the function takes in the trajectory object
        as the first argument.

        """

        return self.function(traj,  **self.func_args)


class DihedralFeaturizer(Featurizer):
    """Featurizer based on dihedral angles.

    This featurizer transforms a dataset containing MD trajectories into
    a vector dataset by representing each frame in each of the MD trajectories
    by a vector containing one or more of the backbone or side-chain dihedral
    angles, or the sin and cosine of these angles.

    Parameters
    ----------
    types : list
        One or more of ['phi', 'psi', 'omega', 'chi1', 'chi2', 'chi3', 'chi4']
    sincos : bool
        Instead of outputting the angle, return the sine and cosine of the
        angle as separate features.
    """

    def __init__(self, types=['phi', 'psi'], sincos=True):
        if isinstance(types, str):
            types = [types]
        self.types = list(types)  # force a copy
        self.sincos = sincos

        known = {'phi', 'psi', 'omega', 'chi1', 'chi2', 'chi3', 'chi4'}
        if not set(types).issubset(known):
            raise ValueError('angles must be a subset of %s. you supplied %s' %
                             (str(known), str(types)))

    def describe_features(self, traj):
        """Return a list of dictionaries describing the dihderal features.

        Parameters
        ----------
        traj : mdtraj.Trajectory
            The trajectory to describe

        Returns
        -------
        feature_descs : list of dict
            Dictionary describing each feature with the following information
            about the atoms participating in each dihedral
                - resnames: unique names of residues
                - atominds: the four atom indicies
                - resseqs: unique residue sequence ids (not necessarily
                  0-indexed)
                - resids: unique residue ids (0-indexed)
                - featurizer: Dihedral
                - featuregroup: the type of dihedral angle and whether sin or
                  cos has been applied.
        """

        feature_descs = []
        for dihed_type in self.types:
            # TODO: Don't recompute dihedrals, just get the indices
            func = getattr(md, 'compute_%s' % dihed_type)
            # ainds is a list of four-tuples of atoms participating
            # in each dihedral
            aind_tuples, _ = func(traj)
            top = traj.topology
            zippy = zippy_maker(aind_tuples, top)

            if self.sincos:
                zippy = itertools.product(['Dihedral'],[dihed_type], ['sin', 'cos'], zippy)
            else:
                zippy = itertools.product(['Dihedral'],[dihed_type], ['nosincos'], zippy)

            feature_descs.extend(dict_maker(zippy))

        return feature_descs

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
            _, y = func(traj)

            if self.sincos:
                x.extend([np.sin(y), np.cos(y)])
            else:
                x.append(y)

        return np.hstack(x)


class VonMisesFeaturizer(Featurizer):
    """Featurizer based on dihedral angles soft-binned along the unit circle.

    This featurizer transforms a dataset containing MD trajectories into
    a vector dataset by representing each frame in each of the MD trajectories
    as a vector containing n soft-bins for each dihedral angle. Soft-bins are
    computed by arranging n equal-spaced von Mises distributions along the unit
    circle and using the PDF of those distributions to define the bin value.

    Parameters
    ----------
    types : list
        One or more of ['phi', 'psi', 'omega', 'chi1', 'chi2', 'chi3', 'chi4']
    n_bins : int
        Number of  von Mises distributions to be used.
    kappa : int or float
        Shape parameter for the von Mises distributions.
    """

    def __init__(self, types=['phi', 'psi'], n_bins=18, kappa=20.):
        if isinstance(types, str):
            types = [types]
        self.types = list(types)  # force a copy

        if not isinstance(n_bins, int):
            raise TypeError('bins must be of type int.')
        if not isinstance(kappa, (int, float)):
            raise TypeError('kappa must be numeric.')

        self.loc = np.linspace(0, 2*np.pi, n_bins)
        self.kappa = kappa
        self.n_bins = n_bins

        known = {'phi', 'psi', 'omega', 'chi1', 'chi2', 'chi3', 'chi4'}
        if not set(types).issubset(known):
            raise ValueError('angles must be a subset of %s. you supplied %s' %
                             (str(known), str(types)))

    def describe_features(self, traj):
        """Return a list of dictionaries describing the dihderal features.

        Parameters
        ----------
        traj : mdtraj.Trajectory
            The trajectory to describe

        Returns
        -------
        feature_descs : list of dict
            Dictionary describing each feature with the following information
            about the atoms participating in each dihedral
                - resnames: unique names of residues
                - atominds: the four atom indicies
                - resseqs: unique residue sequence ids (not necessarily
                  0-indexed)
                - resids: unique residue ids (0-indexed)
                - featurizer: Dihedral
                - featuregroup: The bin index(0..nbins-1)
                and dihedral type(phi/psi/chi1 etc )
        """
        feature_descs = []
        for dihed_type in self.types:
            # TODO: Don't recompute dihedrals, just get the indices
            func = getattr(md, 'compute_%s' % dihed_type)
            # ainds is a list of four-tuples of atoms participating
            # in each dihedral
            aind_tuples, _ = func(traj)

            top = traj.topology
            bin_info =[]
            resseqs = []
            resids = []
            resnames = []
            all_aind = []
            #its bin0---all phis bin1--all_phis
            for bin_index in range(self.n_bins):
                for ainds in aind_tuples:
                    resid = set(top.atom(ai).residue.index for ai in ainds)
                    all_aind.append(ainds)
                    bin_info += ["bin-%d"%bin_index]
                    resids += [list(resid)]
                    reseq = set(top.atom(ai).residue.resSeq for ai in ainds)
                    resseqs += [list(reseq)]
                    resname = set(top.atom(ai).residue.name for ai in ainds)
                    resnames += [list(resname)]

            zippy = zip(all_aind, resseqs, resids, resnames)
            #fast check to make sure we have the right number of features
            assert len(bin_info) == len(aind_tuples) * self.n_bins

            zippy = zip(["VonMises"]*len(bin_info),
                        [dihed_type]*len(bin_info),
                        bin_info,
                        zippy)

            feature_descs.extend(dict_maker(zippy))

        return feature_descs


    def partial_transform(self, traj):
        """Featurize an MD trajectory into a vector space via calculation
        of soft-bins over dihdral angle space.

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
            _, y = func(traj)
            res = vm.pdf(y[..., np.newaxis],
                         loc=self.loc, kappa=self.kappa)
            #we reshape the results using a  Fortran-like index order,
            #so that it goes over the columns first. This should put the results
            #phi dihedrals(all bin0 then all bin1), psi dihedrals(all_bin1)
            x.extend(np.reshape(res, (1, -1, self.n_bins*y.shape[1]), order='F'))
        return np.hstack(x)


class AlphaAngleFeaturizer(Featurizer):
    """Featurizer to extract alpha (dihedral) angles.

    The alpha angle of residue `i` is the dihedral formed by the four CA atoms
    of residues `i-1`, `i`, `i+1` and `i+2`.

    Parameters
    ----------
    sincos : bool
        Instead of outputting the angle, return the sine and cosine of the
        angle as separate features.
    """

    def __init__(self, sincos=True):
        self.sincos = sincos
        self.atom_indices = None

    def partial_transform(self, traj):
        """Featurize an MD trajectory into a vector space via calculation
        of dihedral (torsion) angles of alpha carbon backbone

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

        """

        ca = [a.index for a in traj.top.atoms if a.name == 'CA']
        if len(ca) < 4:
            return np.zeros((len(traj), 0), dtype=np.float32)

        alpha_indices = np.array(
            [(ca[i - 1], ca[i], ca[i + 1], ca[i + 2]) for i in range(1, len(ca) - 2)])
        result = md.compute_dihedrals(traj, alpha_indices)

        x = []
        if self.atom_indices is None:
            self.atom_indices = np.vstack(alpha_indices)
        if self.sincos:
            x.extend([np.cos(result), np.sin(result)])
        else:
            x.append(result)
        return np.hstack(x)

    def describe_features(self, traj):
        """Return a list of dictionaries describing the dihderal features.

        Parameters
        ----------
        traj : mdtraj.Trajectory
            The trajectory to describe

        Returns
        -------
        feature_descs : list of dict
            Dictionary describing each feature with the following information
            about the atoms participating in each dihedral
                - resnames: unique names of residues
                - atominds: the four atom indicies
                - resseqs: unique residue sequence ids (not necessarily
                  0-indexed)
                - resids: unique residue ids (0-indexed)
                - featurizer: Alpha Angle
                - featuregroup: the type of dihedral angle and whether sin or
                  cos has been applied.
        """
        feature_descs = []
        # fill in the atom indices using just the first frame
        self.partial_transform(traj[0])
        top = traj.topology
        if self.atom_indices is None:
            raise ValueError("Cannot describe features for "
                             "trajectories with "
                              "fewer than 4 alpha carbon"
                              "using AlphaAngleFeaturizer.")

        aind_tuples = self.atom_indices

        zippy = zippy_maker(aind_tuples, top)

        if self.sincos:
            zippy = itertools.product(["AlphaAngle"], ["N/A"], ['cos', 'sin'], zippy)
        else:
            zippy = itertools.product(["AlphaAngle"], ["N/A"], ['nosincos'], zippy)

        feature_descs.extend(dict_maker(zippy))

        return feature_descs


class KappaAngleFeaturizer(Featurizer):
    """Featurizer to extract kappa angles.

    The kappa angle of residue `i` is the angle formed by the three CA atoms
    of residues `i-2`, `i` and `i+2`. This featurizer extracts the
    `n_residues - 4` kappa angles of each frame in a trajectory.

    Parameters
    ----------
    cos : bool
        Compute the cosine of the angle instead of the angle itself.
    """

    def __init__(self, cos=True):
        self.cos = cos
        self.atom_indices = None

    def partial_transform(self, traj):
        ca = [a.index for a in traj.top.atoms if a.name == 'CA']
        if len(ca) < 5:
            return np.zeros((len(traj), 0), dtype=np.float32)

        angle_indices = np.array(
            [(ca[i - 2], ca[i], ca[i + 2]) for i in range(2, len(ca) - 2)])
        result = md.compute_angles(traj, angle_indices)

        if self.atom_indices is None:
            self.atom_indices = np.vstack(angle_indices)
        if self.cos:
            return np.cos(result)

        assert result.shape == (traj.n_frames, traj.n_residues - 4)
        return result


    def describe_features(self, traj):
        """Return a list of dictionaries describing the dihderal features.

        Parameters
        ----------
        traj : mdtraj.Trajectory
            The trajectory to describe

        Returns
        -------
        feature_descs : list of dict
            Dictionary describing each feature with the following information
            about the atoms participating in each dihedral
                - resnames: unique names of residues
                - atominds: the four atom indicies
                - resseqs: unique residue sequence ids (not necessarily
                  0-indexed)
                - resids: unique residue ids (0-indexed)
                - featurizer: KappaAngle
                - featuregroup: the type of dihedral angle and whether
                  cos has been applied.
        """
        feature_descs = []
        # fill in the atom indices using just the first frame
        self.partial_transform(traj[0])
        top = traj.topology
        if self.atom_indices is None:
            raise ValueError("Cannot describe features for trajectories "
                             "with fewer than 5 alpha carbon"
                             "using KappaAngle Featurizer")
        aind_tuples = self.atom_indices
        zippy = zippy_maker(aind_tuples, top)
        if self.cos:
            zippy = itertools.product(["Kappa"],["N/A"], ['cos'], zippy)
        else:
            zippy = itertools.product(["Kappa"],["N/A"], ['nocos'], zippy)

        feature_descs.extend(dict_maker(zippy))


        return feature_descs



class SASAFeaturizer(Featurizer):
    """Featurizer based on solvent-accessible surface areas.

    Parameters
    ----------

    mode : {'atom', 'residue'}, default='residue'
        In mode == 'atom', the extracted features are the per-atom
        SASA. In mode == 'residue', this is consolidated down to
        the per-residue SASA by summing over the atoms in each
        residue.

    Other Parameters
    ----------------
    probe_radius : float
    n_sphere_points : int
        If supplied, these arguments will be passed directly to
        `mdtraj.shrake_rupley`, overriding default values.

    See Also
    --------
    mdtraj.shrake_rupley
    """

    def __init__(self, mode='residue', **kwargs):
        self.mode = mode
        self.kwargs = kwargs

    def partial_transform(self, traj):
        return md.shrake_rupley(traj, mode=self.mode, **self.kwargs)


class ContactFeaturizer(Featurizer):
    """Featurizer based on residue-residue distances.

    This featurizer transforms a dataset containing MD trajectories into
    a vector dataset by representing each frame in each of the MD trajectories
    by a vector of the distances between pairs of amino-acid residues.

    The exact method for computing the the distance between two residues
    is configurable with the ``scheme`` parameter.

    Parameters
    ----------
    contacts : np.ndarray or 'all'
        array containing (0-indexed) indices of the residues to compute the
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

    def _transform(self, distances):
        return distances

    def partial_transform(self, traj):
        """Featurize an MD trajectory into a vector space derived from
        residue-residue distances

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

        distances, _ = md.compute_contacts(traj, self.contacts,
                                           self.scheme, self.ignore_nonprotein)
        return self._transform(distances)


    def describe_features(self, traj):
        """Return a list of dictionaries describing the contacts features.

        Parameters
        ----------
        traj : mdtraj.Trajectory
            The trajectory to describe

        Returns
        -------
        feature_descs : list of dict
            Dictionary describing each feature with the following information
            about the atoms participating in each dihedral
                - resnames: unique names of residues
                - atominds: the four atom indicies
                - resseqs: unique residue sequence ids (not necessarily
                  0-indexed)
                - resids: unique residue ids (0-indexed)
                - featurizer: Contact
                - featuregroup: ca, heavy etc.
        """
        feature_descs = []
        # fill in the atom indices using just the first frame
        distances, residue_indices = md.compute_contacts(traj[0], self.contacts,
                                                         self.scheme,
                                                         self.ignore_nonprotein
                                                         )
        top = traj.topology

        aind = []
        resseqs = []
        resnames = []
        for resid_ids in residue_indices:
            aind += ["N/A"]
            resseqs += [[top.residue(ri).resSeq for ri in resid_ids]]
            resnames += [[top.residue(ri).name for ri in resid_ids]]

        zippy = itertools.product(["Contact"], [self.scheme],
                                  ["Ignore_Protein {}".format(self.ignore_nonprotein)],
                                  zip(aind, resseqs, residue_indices, resnames))

        feature_descs.extend(dict_maker(zippy))

        return feature_descs


class BinaryContactFeaturizer(ContactFeaturizer):
    """Featurizer based on residue-residue contacts below a cutoff.

    This featurizer transforms a dataset containing MD trajectories into
    a vector dataset by representing each frame in each of the MD trajectories
    by a vector of the binary contacts between pairs of amino-acid residues.

    The exact method for computing the the distance between two residues
    is configurable with the ``scheme`` parameter.

    Parameters
    ----------
    contacts : np.ndarray or 'all'
        array containing (0-indexed) indices of the residues to compute the
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
    cutoff : float, default=0.8
        Distances shorter than CUTOFF [nm] are returned as '1' and
        distances longer than CUTOFF are returned as '0'.
    """

    def __init__(self, contacts='all', scheme='closest-heavy', ignore_nonprotein=True, cutoff=0.8):
        super(BinaryContactFeaturizer, self).__init__(contacts=contacts, scheme=scheme,
                                                    ignore_nonprotein=ignore_nonprotein)
        if cutoff < 0:
            raise ValueError('cutoff must be a positive distance [nm]')
        self.cutoff = cutoff

    def _transform(self, distances):
        return distances < self.cutoff


class LogisticContactFeaturizer(ContactFeaturizer):
    """Featurizer based on logistic-transformed residue-residue contacts.

    This featurizer transforms a dataset containing MD trajectories into
    a vector dataset by representing each frame in each of the MD trajectories
    by a vector of the distances between pairs of amino-acid residues transformed
    by the logistic function (reflected across the x axis):

    result = [1 + exp(k*(distances - cutoff))]^-1

    The exact method for computing the the distance between two residues
    is configurable with the ``scheme`` parameter.

    Parameters
    ----------
    contacts : np.ndarray or 'all'
        array containing (0-indexed) indices of the residues to compute the
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
    center : float, default=0.8
        Determines the midpoint of the sigmoid, x_0, [nm]. Distances
        shorter than CENTER will return values greater than 0.5 and
        distances larger than CENTER will return values smaller than 0.5.
    steepness : float, default=20
        Determines the steepness of the logistic curve, [1/nm]. Small 
        and large distances will approach ouput values of 1 and 0,
        respectively, more quickly.
    """


    def __init__(self, contacts='all', scheme='closest-heavy', ignore_nonprotein=True,
                                                            center=0.8, steepness=20):
        super(LogisticContactFeaturizer, self).__init__(contacts=contacts, scheme=scheme,
                                                    ignore_nonprotein=ignore_nonprotein)
        if center < 0:
            raise ValueError('center must be a positive distance [nm]')
        if steepness < 0:
            raise ValueError('steepness must be a positive inverse distance [1/nm]')

        self.center = center
        self.steepness = steepness

    def _transform(self, distances):
        result = 1.0/(1+np.exp(self.steepness*(distances-self.center)))
        return result


class GaussianSolventFeaturizer(Featurizer):
    """Featurizer on weighted pairwise distance between solute and solvent.

    We apply a Gaussian kernel to each solute-solvent pairwise distance
    and sum the kernels for each solute atom, resulting in a vector
    of len(solute_indices).

    The values can be physically interpreted as the degree of solvation
    of each solute atom.

    Parameters
    ----------
    solute_indices : np.ndarray, shape=(n_solute,)
        Indices of solute atoms
    solvent_indices : np.ndarray, shape=(n_solvent,)
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
        self.solute_indices = solute_indices
        self.solvent_indices = solvent_indices
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
    """Featurize an MD trajectory into a vector space with the raw
    cartesian coordinates

    Parameters
    ----------
    atom_indices : None or array-like, dtype=int, shape=(n_atoms)
        If specified, only return the coordinates for the atoms
        given by atom_indices. Otherwise return all atoms
    ref_traj : None or md.Trajectory
        If specified, superpose each trajectory to the first frame of
        ref_traj before getting positions. If atom_indices is also
        specified, only superpose based on those atoms. The superposition
        will modify each transformed trajectory *in place*.

    """

    def __init__(self, atom_indices=None, ref_traj=None):
        super(RawPositionsFeaturizer, self).__init__()

        self.atom_indices = atom_indices

        if atom_indices is not None and ref_traj is not None:
            self.ref_traj = ref_traj.atom_slice(atom_indices)
        else:
            self.ref_traj = ref_traj

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

        Notes
        -----
        If you requested superposition (gave `ref_traj` in __init__) the
        input trajectory will be modified.

        See Also
        --------
        transform : simultaneously featurize a collection of MD trajectories
        """
        # Optionally take only certain atoms
        if self.atom_indices is not None:
            p_traj = traj.atom_slice(self.atom_indices)
        else:
            p_traj = traj

        # Optionally superpose to a reference trajectory.
        if self.ref_traj is not None:
            p_traj.superpose(self.ref_traj, parallel=False)

        # Get the positions and reshape.
        value = p_traj.xyz.reshape(len(p_traj), -1)
        return value


class DRIDFeaturizer(Featurizer):
    """Featurizer based on distribution of reciprocal interatomic
    distances (DRID)

    This featurizer transforms a dataset containing MD trajectories into
    a vector dataset by representing each frame in each of the MD trajectories
    by a vector containing the first three moments of a collection of
    reciprocal interatomic distances. For details, see [1].

    References
    ----------
    .. [1] Zhou, Caflisch; Distribution of Reciprocal of Interatomic Distances:
       A Fast Structural Metric. JCTC 2012 doi:10.1021/ct3003145

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


class TrajFeatureUnion(BaseEstimator, sklearn.pipeline.FeatureUnion):
    """sklearn.pipeline.FeatureUnion adapted for multiple sequences
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
            delayed(sklearn.pipeline._transform_one)(trans, name, traj_list,
                                                     self.transformer_weights)
            for name, trans in self.transformer_list)

        X_i_stacked = [np.hstack([Xs[feature_ind][trj_ind]
                       for feature_ind in range(len(Xs))])
                       for trj_ind in range(len(Xs[0]))]

        return X_i_stacked


class Slicer(Featurizer):
    """Extracts slices (e.g. subsets) from data along the feature dimension.

    Parameters
    ----------
    index : array_like of integer, optional
        If given, extract only these features by index. This corresponds
        to selecting these columns from the feature-trajectories.
    first : int, optional
        If given, extract the first this-many features. This is useful
        when features are sorted like in PCA or tICA.

    Notes
    -----
    You must give either index or first (but not both)

    """

    def __init__(self, index=None, first=None):

        if index is None and first is None:
            raise ValueError("Please specify either index or first, "
                             "not neither")
        if index is not None and first is not None:
            raise ValueError("Please specify either index or first, "
                             "not both.")

        self.index = index
        self.first = first

    def partial_transform(self, traj):
        """Slice a single input array along to select a subset of features.

        Parameters
        ----------
        traj : np.ndarray, shape=(n_samples, n_features)
            A sample to slice.

        Returns
        -------
        sliced_traj : np.ndarray shape=(n_samples, n_feature_subset)
            Slice of traj
        """
        if self.index is not None:
            return traj[:, self.index]
        else:
            return traj[:, :self.first]


class FirstSlicer(object):

    def __init__(self, *args, **kwargs):
        raise NotImplementedError("Please use Slicer(first=x)")
