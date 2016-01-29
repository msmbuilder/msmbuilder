# Author: Kyle A. Beauchamp <kyleabeauchamp@gmail.com>
# Contributors: Robert McGibbon <rmcgibbo@gmail.com>,
#               Matthew Harrigan <matthew.p.harrigan@gmail.com>
#               Brooke Husic <brookehusic@gmail.com>
# Copyright (c) 2015, Stanford University and the Authors
# All rights reserved.

#-----------------------------------------------------------------------------
# Imports
#-----------------------------------------------------------------------------
from __future__ import print_function, division, absolute_import

from six.moves import cPickle
import numpy as np
import mdtraj as md
from sklearn.base import TransformerMixin
import sklearn.pipeline
from sklearn.externals.joblib import Parallel, delayed
from msmbuilder import libdistance

from ..base import BaseEstimator

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

    def save(self, filename):
        with open(filename, 'wb') as f:
            cPickle.dump(self, f)


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
        traj.superpose(self.reference_traj, atom_indices=self.superpose_atom_indices)
        diff2 = (traj.xyz[:, self.atom_indices] -
                 self.reference_traj.xyz[0, self.atom_indices]) ** 2
        x = np.sqrt(np.sum(diff2, axis=2))
        return x



class StrucRMSDFeaturizer(Featurizer):
    """Featurizer based on RMSD to one or more reference structures.

    This featurizer inputs a trajectory to be analyzed ('traj') and a
    reference trajectory ('ref') and outputs the RMSD of each frame of
    traj with respect to each frame in ref. The output is a numpy array
    with n_rows = traj.n_frames and n_columns = ref.n_frames.

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
        result = libdistance.cdist(traj, self.reference_traj, 'rmsd')
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
        d = md.geometry.compute_distances(traj, self.pair_indices, periodic=self.periodic)
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
    *args : list
        arguments to pass to the function MINUS the trajectory keyword
    **kwargs: kwargs
        key word arguments to pass to the function MINUS the trajectory
    keyword

    UseCase:
    ---------
    function = compute_dihedrals
    f = FunctionFeaturizer(function, indices=[[0,1,2,3]])
    results = f.transform(dataset)

    Notes
    ----------
    This Featurizer assumes that the function takes in the trajectory object
    as the first argument
    """

    def __init__(self, function, *args, **kwargs):
        if hasattr(function, '__call__'):
            self.function = function
            self.args = args
            self.kwargs = kwargs
        else:
            raise Exception("Sorry but we "
                            "couldn't use the provided "
                            "function.")

    def partial_transform(self, traj):
        """Featurize a MD trajectory into a vector by
        applying the given function unto the trajectory.

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

         Notes:
        --------
         This method assumes that the function takes in the trajectory object
        as the first argument.

        """
        x = []
        x.append(self.function(traj, *self.args, **self.kwargs))
        return np.hstack(x)

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
            raise ValueError('angles must be a subset of %s. you supplied %s' % (
                str(known), str(types)))

    def describe_features(self, traj):
        """Return a list of dictionaries describing the Dihderal features."""
        x = []
        for a in self.types:
            func = getattr(md, 'compute_%s' % a)
            aind, y = func(traj)
            n = len(aind)

            resSeq = [(np.unique([traj.top.atom(j).residue.resSeq for j in i])) for i in aind]
            resid = [(np.unique([traj.top.atom(j).residue.index for j in i])) for i in aind]
            resnames = [[traj.topology.residue(j).name for j in i ] for i in resid]


            bigclass = ["dihedral"] * n
            smallclass = [a] * n

            if self.sincos:
                #x.extend([np.sin(y), np.cos(y)])
                aind =  list(aind) * 2
                resnames = resnames * 2
                resSeq = resSeq * 2
                resid = resid * 2
                otherInfo = (["sin"] * n) + (["cos"] * n)
                bigclass = bigclass * 2
                smallclass = smallclass * 2
            else:
                otherInfo = ["nosincos"] * n

            for i in range(len(resnames)):
                d_i = dict(resname=resnames[i], atomind=aind[i],resSeq=resSeq[i], resid=resid[i],\
                           otherInfo=otherInfo[i], bigclass=bigclass[i], smallclass=smallclass[i])
                x.append(d_i)

        return x

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
            _,y = func(traj)
            if self.sincos:
                x.extend([np.sin(y), np.cos(y)])
            else:
                x.append(y)
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
            [(ca[i - 1], ca[i], ca[i+1], ca[i + 2]) for i in range(1, len(ca) - 2)])
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
        """Return a list of dictionaries describing the alpha dihedral angle features."""
        x = []
        #fill in the atom indices using just the first frame
        res_ = self.partial_transform(traj[0])
        if self.atom_indices is not None:
            aind = self.atom_indices
            n = len(aind)
            resSeq = [(np.unique([traj.top.atom(j).residue.resSeq for j in i])) for i in aind]
            resid = [(np.unique([traj.top.atom(j).residue.index for j in i])) for i in aind]
            resnames = [[traj.topology.residue(j).name for j in i ] for i in resid]
            bigclass = ["dihedral"] * n
            smallclass = ["alpha"] * n

            if self.sincos:
                #x.extend([np.sin(y), np.cos(y)])
                aind =  list(aind) * 2
                resnames = resnames * 2
                resSeq = resSeq * 2
                resid = resid * 2
                otherInfo = (["sin"] * n) + (["cos"] * n)
                bigclass = bigclass * 2
                smallclass = smallclass * 2
            else:
                otherInfo = ["nosincos"] * n

            for i in range(len(resnames)):
                d_i = dict(resname=resnames[i], atomind=aind[i],resSeq=resSeq[i], resid=resid[i],\
                           otherInfo=otherInfo[i], bigclass=bigclass[i], smallclass=smallclass[i])
                x.append(d_i)

            return x
        else:
            raise UserWarning("Cannot describe features for trajectories with fewer than 4 alpha carbon\
                              using AlphaAngleFeaturizer")



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
        """Return a list of dictionaries describing the Kappa angle features."""
        x = []
        #fill in the atom indices using just the first frame
        res_ = self.partial_transform(traj[0])
        if self.atom_indices is not None:
            aind = self.atom_indices
            n = len(aind)
            resSeq = [(np.unique([traj.top.atom(j).residue.resSeq for j in i])) for i in aind]
            resid = [(np.unique([traj.top.atom(j).residue.index for j in i])) for i in aind]
            resnames = [[traj.topology.residue(j).name for j in i ] for i in resid]
            bigclass = ["angle"] * n
            smallclass = ["kappa"] * n

            if self.cos:
                otherInfo = (["cos"] * n)
            else:
                otherInfo = ["nocos"] * n

            assert len(self.atom_indices)==len(resnames)

            for i in range(len(resnames)):
                    d_i = dict(resname=resnames[i], atomind=aind[i],resSeq=resSeq[i], resid=resid[i],\
                               otherInfo=otherInfo[i], bigclass=bigclass[i], smallclass=smallclass[i])
                    x.append(d_i)

            return x
        else:
            raise UserWarning("Cannot describe features for trajectories with fewer than 5 alpha carbon\
                              using KappaAngle Featurizer")


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
    """Featurizer based on residue-residue distances

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

    def describe_features(self, traj):
        """Return a list of dictionaries describing the features in Contacts."""
        x = []
        #fill in the atom indices using just the first frame
        distances,residue_indices = md.compute_contacts(traj, self.contacts, self.scheme, self.ignore_nonprotein)
        n = residue_indices.shape[0]
        aind = ["N/A"] * n
        resSeq = [np.array([traj.top.residue(j).resSeq for j in i]) for i in residue_indices]
        resid = [np.array([traj.top.residue(j).index for j in i]) for i in residue_indices]
        resnames = [[traj.topology.residue(j).name for j in i ] for i in resid]
        bigclass = [self.contacts] * n
        smallclass = [self.scheme] * n
        otherInfo = [self.ignore_nonprotein]*n

        for i in range(n):
            d_i = dict(resname=resnames[i], atomind=aind[i],resSeq=resSeq[i], resid=resid[i],\
                               otherInfo=otherInfo[i], bigclass=bigclass[i], smallclass=smallclass[i])
            x.append(d_i)

        return x



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


class Slicer(Featurizer):
    """Extracts slices (e.g. subsets) from data along the feature dimension.

    Parameters
    ----------
    index : list of integers, optional, default=None
        These indices are the feature indices that will be selected
        by the Slicer.transform() function.  

    """

    def __init__(self, index=None):
        self.index = index

    def partial_transform(self, X):
        """Slice a single input array along to select a subset of features.

        Parameters
        ----------
        X : np.ndarray, shape=(n_samples, n_features)
            A sample to slice.

        Returns
        -------
        X2 : np.ndarray shape=(n_samples, n_feature_subset)
            Slice of X
        """
        return X[:, self.index]


class FirstSlicer(Slicer):
    """Extracts slices (e.g. subsets) from data along the feature dimension.

    Parameters
    ----------
    first : int, optional, default=None
        Select the first N features.  This is essentially a shortcut for
        `Slicer(index=arange(first))`

    """

    def __init__(self, first=None):
        self.first = first
    
    @property
    def index(self):
        return np.arange(self.first)
