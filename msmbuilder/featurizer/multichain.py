# Author: Brooke Husic <brookehusic@gmail.com>
# Contributors: 
# Copyright (c) 2016, Stanford University and the Authors
# All rights reserved.

from __future__ import print_function, division, absolute_import

import warnings

import mdtraj as md
from mdtraj.utils import ensure_type
from mdtraj.utils.six import string_types
from mdtraj.utils.six.moves import xrange
import numpy as np
import itertools

from msmbuilder import libdistance
from . import Featurizer


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


class LigandFeaturizer(Featurizer):
    """Base class for objects that featurize protein-ligand trajectories.

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

        self.contacts = self._guess_chains(reference_frame)

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
        single-frame conformation if binding pocket is specified
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
        super(LigandContactFeaturizer, self).__init__(protein_chain=protein_chain,
                                                      ligand_chain=ligand_chain,
                                                      reference_frame=reference_frame)
        self.contacts = contacts
        self.scheme = scheme
        self.binding_pocket = binding_pocket

        self.contacts = self._get_contact_pairs(reference_frame, contacts)

    def _get_contact_pairs(self, traj, contacts):
        if self.scheme=='ca':
            if not any(a for a in traj.top.chain(ligand_chain).atoms
                       if a.name.lower() == 'ca'):
                raise ValueError("Bad scheme: the ligand has no alpha carbons")

        # this is really similar to mdtraj/contact.py, but ensures that md.compute_contacts
        # is always seeing an array of exactly the contacts we want to specify
        if isinstance(contacts, string_types):
            if contacts.lower() != 'all':
                raise ValueError('(%s) is not a valid contacts specifier' % contacts.lower())

            residue_pairs = []
            p_offset = 0
            for chain in range(0, self.protein_chain):
                p_offset += traj.top.chain(chain).n_residues
            l_offset = 0
            for chain in range(0, self.ligand_chain):
                l_offset += traj.top.chain(chain).n_residues
            for i in xrange(traj.top.chain(self.protein_chain).n_residues):
                for j in xrange(traj.top.chain(self.ligand_chain).n_residues):
                    residue_pairs.append((i+p_offset, j+l_offset))

            residue_pairs = np.array(residue_pairs)

            if len(residue_pairs) == 0:
                raise ValueError('No acceptable residue pairs found')

        else:
            residue_pairs = ensure_type(np.asarray(contacts), dtype=np.int, ndim=2, name='contacts',
                                                   shape=(None, 2), warn_on_cast=False)
            if not np.all((residue_pairs >= 0) * (residue_pairs < traj.n_residues)):
                raise ValueError('contacts requests a residue that is not in the permitted range')

        if self.binding_pocket is not 'all':
            ref_distances, _ = md.compute_contacts(traj,
                                     residue_pairs, self.scheme,
                                     ignore_nonprotein=False)
            residue_pairs = residue_pairs[np.where(ref_distances<
                                     self.binding_pocket)[1]]

        return residue_pairs

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
        distances, residue_indices = md.compute_contacts(traj[0], self.contacts,
                                                         self.scheme,
                                                         ignore_nonprotein=False
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
        single-frame conformation if binding pocket is specified
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
        super(BinaryLigandContactFeaturizer, self).__init__(protein_chain=protein_chain,
                ligand_chain=ligand_chain, reference_frame=reference_frame,
                contacts=contacts, scheme=scheme, binding_pocket=binding_pocket)

        if cutoff < 0:
            raise ValueError('cutoff must be a positive distance [nm]')
        self.cutoff = cutoff

    def _transform(self, distances):
        return distances < self.cutoff

