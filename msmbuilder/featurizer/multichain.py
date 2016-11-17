# Author: Brooke Husic <brookehusic@gmail.com>
# Contributors: 
# Copyright (c) 2016, Stanford University and the Authors
# All rights reserved.

from __future__ import print_function, division, absolute_import

import warnings

import mdtraj as md
from mdtraj.utils import ensure_type
from mdtraj.utils.six import string_types
import numpy as np
import itertools
import warnings

from msmbuilder import libdistance
from . import Featurizer, zippy_maker, dict_maker, featurize_all


class LigandFeaturizer(Featurizer):
    """Base class for objects that featurize protein-ligand trajectories.

    Parameters
    ----------
    protein_chain : int or 'auto', default='auto'
        chain in the trajectory containing the protein of interest. 'auto'
        chooses the longest alpha-carbon containing chain. for chains with
        an identical number of atoms, 'auto' will choose the first chain
        in the topology file.
    ligand_chain : int or 'auto', default='auto'
        chain in the trajectory containing the ligand of interest. 'auto'
        chooses the chain containing the most atoms, not to exceed 100
        atoms, that is not already the protein chain. for chains with
        an identical number of atoms, 'auto' will choose the first chain
        in the topology file.
    reference_frame : md.Trajectory, default=None
        single-frame conformation to get chain information; also defines
        the binding pocket if specified

    Notes
    -----
    At the bare minimum, a featurizer must implement the `partial_transform(traj)`
    member function.  A `transform(traj_list)` for featurizing multiple
    trajectories in batch will be provided.
    """


    def __init__(self, protein_chain='auto', ligand_chain='auto',
                 reference_frame=None):
        self.protein_chain = protein_chain
        self.ligand_chain = ligand_chain
        self.reference_frame = reference_frame

        if reference_frame is None:
            raise ValueError("Please specify a reference frame")

        if reference_frame.n_frames is not 1:
            raise ValueError("Reference frame must be a single frame")

        if reference_frame.top.n_chains < 2:
            raise ValueError("Topology must have at least two chains")

        self.contacts = self._guess_chains(self.reference_frame)
        self.p_residue_offset = self._get_residue_offset(self.protein_chain)
        self.l_residue_offset = self._get_residue_offset(self.ligand_chain)
        self.p_atom_offset = self._get_atom_offset(self.protein_chain)
        self.l_atom_offset = self._get_atom_offset(self.ligand_chain)


    def _guess_chains(self, traj):
        if self.protein_chain == 'auto' or self.ligand_chain == 'auto':
            chain_dict = {}
        for i in range(traj.top.n_chains):
            # each entry in the chain_dict is a list:
            # [number of atoms, has a CA, has <100 atoms]
            chain_dict.update({i: [traj.top.chain(i).n_atoms,
                               any(a for a in traj.top.chain(i).atoms
                                   if a.name.lower() == 'ca'),
                               traj.top.chain(i).n_atoms < 100]})
        # the protein is the CA-containing chain with the most atoms
        if self.protein_chain == 'auto':
            self.protein_chain = max(chain_dict, key=lambda x:
                                     np.prod([chain_dict[x][0],
                                     chain_dict[x][1]]))
        for i in range(traj.top.n_chains):
            if i == self.protein_chain:
                pass
            else:
                chain_dict.update({i: [chain_dict[i][0],
                                   False, chain_dict[i][2]]})
        # the ligand is the chain that is not already the protein chain
        # containing the most atoms up to 99 atoms
        if self.ligand_chain == 'auto':
                self.ligand_chain = max(chain_dict, key = lambda x: 
                                        np.prod([chain_dict[x][0],
                                        not(chain_dict[x][1]),
                                        chain_dict[x][2]]))

    def _get_residue_offset(self, chain_index):
        return self.reference_frame.top.chain(chain_index).residue(0).index


    def _get_atom_offset(self, chain_index):
        return self.reference_frame.top.chain(chain_index).atom(0).index


class LigandContactFeaturizer(LigandFeaturizer):
    """Featurizer based on ligand-protein distances.

    This featurizer transforms a dataset containing MD trajectories into
    a vector dataset by representing each frame in each of the MD trajectories
    by a vector of the distances between pairs of atoms where each pair
    contains one ligand atom and one protein atom.

    The exact method for computing the the distance between two residues
    is configurable with the ``scheme`` parameter.

    Parameters
    ----------
    protein_chain : int or 'auto', default='auto'
        chain in the trajectory containing the protein of interest. 'auto'
        chooses the longest alpha-carbon containing chain. for chains with
        an identical number of atoms, 'auto' will choose the first chain
        in the topology file.
    ligand_chain : int or 'auto', default='auto'
        chain in the trajectory containing the ligand of interest. 'auto'
        chooses the chain containing the most atoms, not to exceed 100
        atoms, that is not already the protein chain. for chains with
        an identical number of atoms, 'auto' will choose the first chain
        in the topology file.
    reference_frame : md.Trajectory, default=None
        single-frame conformation to get chain information; also defines
        the binding pocket if specified
    contacts : np.ndarray or 'all'
        array containing (0-indexed) indices of the residues to compute the
        contacts for. (e.g. np.array([[0, 10], [0, 11]]) would compute
        the contact between residue 0 and residue 10 as well as
        the contact between residue 0 and residue 11.) [NOTE: if no
        array is passed then 'all' contacts are calculated. This means
        that the result will contain contacts between all protein
        residues and the ligand.]
    scheme : {'ca', 'closest', 'closest-heavy'}, default='closest-heavy'
        scheme to determine the distance between two residues:
            'ca' : distance between two residues is given by the distance
                between their alpha carbons. Only allowed if the ligand
                contains an alpha carbon.
            'closest' : distance is the closest distance between any
                two atoms in the residues
            'closest-heavy' : distance is the closest distance between
                any two non-hydrogen atoms in the residues
    binding_pocket : float or 'all', default='all'
        nanometer cutoff to define a binding pocket; if defined, only
        protein atoms within the threshold distance according to the
        topology file will be included

    """

    def __init__(self, protein_chain='auto', ligand_chain='auto',
                 reference_frame=None, contacts='all', 
                 scheme='closest-heavy', binding_pocket='all'):
        super(LigandContactFeaturizer, self).__init__(
                    protein_chain=protein_chain, ligand_chain=ligand_chain,
                    reference_frame=reference_frame)
        self.contacts = contacts
        self.scheme = scheme
        self.binding_pocket = binding_pocket

        self.contacts = self._get_contact_pairs(contacts)

    def _get_contact_pairs(self, contacts):
        if self.scheme=='ca':
            if not any(a for a in self.reference_frame.top.chain(ligand_chain).atoms
                       if a.name.lower() == 'ca'):
                raise ValueError("Bad scheme: the ligand has no alpha carbons")

        # this is really similar to mdtraj/contact.py, but ensures that
        # md.compute_contacts  is always seeing an array of exactly the
        # contacts we want to specify
        if isinstance(contacts, string_types):
            if contacts.lower() != 'all':
                raise ValueError('({}) is not a valid contacts specifier'.format(contacts.lower()))

            self.residue_pairs = []
            for i in np.arange(self.reference_frame.top.chain(self.protein_chain).n_residues):
                for j in np.arange(self.reference_frame.top.chain(self.ligand_chain).n_residues):
                    self.residue_pairs.append((i+self.p_residue_offset,
                                          j+self.l_residue_offset))

            self.residue_pairs = np.array(self.residue_pairs)

            if len(self.residue_pairs) == 0:
                raise ValueError('No acceptable residue pairs found')

        else:
            self.residue_pairs = ensure_type(np.asarray(contacts),
                                        dtype=np.int, ndim=2, name='contacts',
                                        shape=(None, 2), warn_on_cast=False)
            if not np.all((self.residue_pairs >= 0) *
                          (self.residue_pairs < self.reference_frame.n_residues)): 
                raise ValueError('contacts requests a residue that is not '\
                                 'in the permitted range')

        if self.binding_pocket is not 'all':
            ref_distances, _ = md.compute_contacts(self.reference_frame, 
                                     self.residue_pairs, self.scheme,
                                     ignore_nonprotein=False)
            self.residue_pairs = self.residue_pairs[np.where(ref_distances<
                                     self.binding_pocket)[1]]
            if len(self.residue_pairs) == 0:
                raise ValueError('No residue pairs within binding pocket')

        return self.residue_pairs

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

        # check to make sure topologies are consistent with the reference frame
        try:
            assert traj.top == self.reference_frame.top
        except:
            warnings.warn("The topology of the trajectory is not" +
                          "the same as that of the reference frame," +
                          "which might give meaningless results.")
        distances, _ = md.compute_contacts(traj, self.contacts,
                                        self.scheme, ignore_nonprotein=False)
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
        distances, residue_indices = md.compute_contacts(traj[0],
                                        self.contacts, self.scheme,
                                        ignore_nonprotein=False)
        top = traj.topology

        aind = []
        resseqs = []
        resnames = []
        for resid_ids in residue_indices:
            aind += ["N/A"]
            resseqs += [[top.residue(ri).resSeq for ri in resid_ids]]
            resnames += [[top.residue(ri).name for ri in resid_ids]]

        zippy = itertools.product(["Ligand Contact"], [self.scheme],
                                  ["N/A"],
                                  zip(aind, resseqs, residue_indices, resnames))

        feature_descs.extend(dict_maker(zippy))

        return feature_descs


class BinaryLigandContactFeaturizer(LigandContactFeaturizer):
    """Featurizer based on binary ligand-protein contacts.

    This featurizer transforms a dataset containing MD trajectories into
    a vector dataset by representing each frame in each of the MD trajectories
    by a vector of the binary contacts between pairs of atoms where each pair
    contains one ligand atom and one protein atom.

    The exact method for computing the the distance between two residues
    is configurable with the ``scheme`` parameter.

    Parameters
    ----------
    protein_chain : int or 'auto', default='auto'
        chain in the trajectory containing the protein of interest. 'auto'
        chooses the longest alpha-carbon containing chain. for chains with
        an identical number of atoms, 'auto' will choose the first chain
        in the topology file.
    ligand_chain : int or 'auto', default='auto'
        chain in the trajectory containing the ligand of interest. 'auto'
        chooses the chain containing the most atoms, not to exceed 100
        atoms, that is not already the protein chain. for chains with
        an identical number of atoms, 'auto' will choose the first chain
        in the topology file.
    reference_frame : md.Trajectory, default=None
        single-frame conformation to get chain information; also defines
        the binding pocket if specified
    contacts : np.ndarray or 'all'
        array containing (0-indexed) indices of the residues to compute the
        contacts for. (e.g. np.array([[0, 10], [0, 11]]) would compute
        the contact between residue 0 and residue 10 as well as
        the contact between residue 0 and residue 11.) [NOTE: if no
        array is passed then 'all' contacts are calculated. This means
        that the result will contain contacts between all protein
        residues and the ligand.]
    scheme : {'ca', 'closest', 'closest-heavy'}, default='closest-heavy'
        scheme to determine the distance between two residues:
            'ca' : distance between two residues is given by the distance
                between their alpha carbons. Only allowed if the ligand
                contains an alpha carbon.
            'closest' : distance is the closest distance between any
                two atoms in the residues
            'closest-heavy' : distance is the closest distance between
                any two non-hydrogen atoms in the residues
    binding_pocket : float or 'all', default='all'
        nanometer cutoff to define a binding pocket; if defined, only
        protein atoms within the threshold distance according to the
        topology file will be included
    cutoff : float, default=0.8
        Distances shorter than CUTOFF [nm] are returned as '1' and
        distances longer than CUTOFF are returned as '0'.
    """


    def __init__(self, protein_chain='auto', ligand_chain='auto',
                 reference_frame=None, contacts='all', 
                 scheme='closest-heavy', binding_pocket='all',
                 cutoff=0.8):
        super(BinaryLigandContactFeaturizer, self).__init__(
                protein_chain=protein_chain, ligand_chain=ligand_chain,
                reference_frame=reference_frame, contacts=contacts,
                scheme=scheme, binding_pocket=binding_pocket)

        if cutoff < 0:
            raise ValueError('cutoff must be a positive distance [nm]')
        self.cutoff = cutoff

    def _transform(self, distances):
        return distances < self.cutoff


class LigandRMSDFeaturizer(LigandFeaturizer):
    """Featurizer based on RMSD to one or more reference structures.

    This featurizer inputs a trajectory to be analyzed ('traj') and a
    reference trajectory ('ref') and outputs the RMSD of each frame of
    traj calculated for a specified set of atom indices after aligning
    the structures based on a different, specified set of atom indices.

    Parameters
    ----------
    protein_chain : int or 'auto', default='auto'
        chain in the trajectory containing the protein of interest. 'auto'
        chooses the longest alpha-carbon containing chain. for chains with
        an identical number of atoms, 'auto' will choose the first chain
        in the topology file.
    ligand_chain : int or 'auto', default='auto'
        chain in the trajectory containing the ligand of interest. 'auto'
        chooses the chain containing the most atoms, not to exceed 100
        atoms, that is not already the protein chain. for chains with
        an identical number of atoms, 'auto' will choose the first chain
        in the topology file.
    reference_frame : md.Trajectory, default=None
        single-frame conformation to get chain information
    reference_traj : md.Trajectory, default=reference_frame
        reference conformation(s) to superpose each frame with respect to
    align_by : {'ligand', 'protein', 'custom'}, default='protein'
        chain on which to align structures. if custom, align_indices
        must be provided.
    align_indices : np.ndarray, shape=(n_atoms,), dtype=int
        the indices of the atoms to superpose with; if not specified,
        all atoms on the 'align_by' chain are used.
    calculate_for : {'ligand', 'protein', 'custom'}, default='ligand'
        chain on which to calculate RMSD. if custom, calculate_indices
        must be provided.
    calculate_indices : np.ndarray, shape=(n_atoms,), dtype=int
        the indices of the atoms to compute the distances with; if not
        specified, all atoms on the 'calculate_for' chain are used.
    """


    def __init__(self, protein_chain='auto', ligand_chain='auto',
                 reference_frame=None, reference_traj=None,
                 align_by='protein', align_indices=None,
                 calculate_for='ligand', calculate_indices=None):
        super(LigandRMSDFeaturizer, self).__init__(
                    protein_chain=protein_chain, ligand_chain=ligand_chain,
                    reference_frame=reference_frame)

        self.reference_traj = reference_traj
        if self.reference_traj is None:
            self.reference_traj = self.reference_frame
        self.n_features = self.reference_traj.n_frames

        if align_by == 'ligand':
            self.align_by = self.ligand_chain
        elif align_by == 'protein':
            self.align_by = self.protein_chain
        elif align_by == 'custom':
            if align_indices is None:
                raise ValueError("Please specify custom align_indices")
        else:
            raise ValueError("Please specify a valid option")

        if calculate_for == 'ligand':
            self.calculate_for = self.ligand_chain
        elif calculate_for == 'protein':
            self.calculate_for = self.protein_chain
        elif calculate_for == 'custom':
            if calculate_indices is None:
                raise ValueError("Please specify custom calculate_indices")
        else:
            raise ValueError("Please specify a valid option")


        if align_indices is not None:
            if align_by is not 'custom':
                if not self._check_indices(self.align_by, align_indices):
                    raise ValueError("align_indices must be on the " \
                                     "align_by chain")
            else:
                if not all(align_indices[i] in
                           range(self.reference_frame.n_atoms)
                           for i in range(len(align_indices))):
                    raise ValueError("align_indices must exist")
            self.align_indices = align_indices
        else:
            self.align_indices = self._get_atom_range(self.align_by)

        if calculate_indices is not None:
            if calculate_for is not 'custom':
                if not self._check_indices(self.calculate_for,
                                           calculate_indices):
                    raise ValueError("calculate_indices must be on the " \
                                     "calculate_for chain")
            else:
                if not all(calculate_indices[i] in
                           range(self.reference_frame.n_atoms)
                           for i in range(len(calculate_indices))):
                    raise ValueError("calculate_indices must exist")
            self.calculate_indices = calculate_indices
        else:
            self.calculate_indices = self._get_atom_range(self.calculate_for)


    # custom option will never see this
    def _get_atom_range(self, chain_index):
        return [a.index for a in self.reference_frame.top.chain(chain_index).atoms]

    def _check_indices(self, chain, indices):
        atom_range = self._get_atom_range(chain)
        return all(indices[i] in atom_range for i in range(len(indices)))


    def _naive_rmsd(self, traj, ref, idx):
        return np.sqrt(np.sum(np.square(traj.xyz[:,idx,:] - ref.xyz[:,idx,:]),
                              axis=(1, 2))/len(idx))


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

        # check to make sure topologies are consistent with the reference frame
        try:
            assert traj.top == self.reference_frame.top
        except:
            warnings.warn("The topology of the trajectory is not" +
                          "the same as that of the reference frame," +
                          "which might give meaningless results.")

        X = np.zeros((traj.n_frames, self.n_features))

        for f in range(self.n_features):
            frame = self.reference_traj[f]
            traj.superpose(frame, atom_indices=self.align_indices)
            X[:,f] = self._naive_rmsd(traj, frame, self.calculate_indices)

        return X
