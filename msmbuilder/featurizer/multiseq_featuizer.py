# Author: Muneeb Sultan <msultan@stanford.edu>
# Copyright (c) 2016, Stanford University and the Authors
# All rights reserved.
from __future__ import print_function, division, absolute_import
import numpy as np
import itertools
from .featurizer import Featurizer, ContactFeaturizer
import warnings
import inspect
import mdtraj as md


class CommonContactFeaturizer(Featurizer):
    """Featurizer based on residue-residue contacts of an alignment file.

    This featurizer will figure out the protein sequence on the fly
    making sure that feature "i" corresponds to the contact distance
    between the same residue indices based upon the given alignment file.

    This featurizer transforms a dataset containing MD trajectories into
    a vector dataset by representing each frame in each of the MD trajectories
    by a vector of the binary contacts between pairs of amino-acid residues.

    The exact method for computing the the distance between two residues
    is configurable with the ``scheme`` parameter.

    The conserved_only flag will only use the wanted_positions if they have
    the same amino acid at that sequence index location.

    Parameters
    ----------
    alignment: list of lists or dictionary
        Fasta formatted alignment. Can either be a list of lists or
        a dictionary keyed on sequence id.
    contacts : np.ndarray or 'all'
        array containing (0-indexed) indices of the alignment to compute the
        contacts for. (e.g. np.array([[0, 10], [0, 11]]) would compute
        the contact between equivalent of residue 0 and residue 10 for
        every protein in the alignment as well as the contact between
        residue 0 and residue 11.) Note that if conserved_only
        flag is True then some of the contacts wont be calculated.
    conserved_only: True or False
        Whether or not to only use conserved(i.e. same aa) residues
        for calculating contacts.
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
    soft_min : bool, default=False
        If soft_min is true, we will use a diffrentiable version of
        the scheme. The exact expression used
         is d = \frac{\beta}{log\sum_i{exp(\frac{\beta}{d_i}})} where
         beta is user parameter which defaults to 20nm. The expression
         we use is copied from the plumed mindist calculator.
         http://plumed.github.io/doc-v2.0/user-doc/html/mindist.html
    soft_min_beta : float, default=20nm
        The value of beta to use for the soft_min distance option.
        Very large values might cause small contact distances to go to 0.
"""

    def __init__(self, alignment=None, contacts='all',
                 same_residue='True',
                 scheme='closest-heavy', ignore_nonprotein=True,
                 soft_min=False, soft_min_beta=20):

        if alignment is None:
            raise ValueError("Common contacts requires an\
                              alignment(either list of lists or dict)")
        if type(alignment) == list:
            self.alignment = {}
            for k, v in enumerate(alignment):
                self.alignment[k] = v
        elif type(alignment) == dict:
            self.alignment = alignment
        else:
            raise ValueError("Alignment is not of type list or dict.")

        self.protein_list = list(self.alignment.keys())
        self.same_residue = same_residue
        self.contacts = contacts
        if contacts is 'all':
            # use the max length(probably a horrible idea)
            max_seq_len = max([len(self.alignment[i])
                               for i in self.alignment.keys()])
            self.contacts = [i for i in itertools.combinations(np.arange(max_seq_len), 2)
                             if abs(i[0] - i[1]) > 3]
            warnings.warn("All valid pair-wise contacts are being calculated")

        self.scheme = scheme
        self.ignore_nonprotein = ignore_nonprotein

        self.all_inv_mappings, \
            self.all_sequences, self.all_res_mappings = self._map_residue_ind_seq_ind(
                self.alignment)

        self.soft_min = soft_min
        self.soft_min_beta = soft_min_beta
        if self.soft_min and not 'soft_min' in inspect.signature(md.compute_contacts).parameters:
            raise ValueError("Sorry but soft_min requires the latest version"
                             "of mdtraj")

        self.feat_dict = self._create_feat_dict()

    def _valid_contact(self, contact):
        seq_ind_i, seq_ind_j = contact
        possible_i_codes = set([self.alignment[p][seq_ind_i] for p in
                                self.alignment.keys()])
        possible_j_codes = set([self.alignment[p][seq_ind_j] for p in
                                self.alignment.keys()])
        # if either of those positions has a addition ignore that contact
        if "-" in possible_i_codes or "-" in possible_j_codes:
            return False
        # if not all residues the same at position i
        elif self.same_residue and len(set(possible_i_codes)) != 1:
            return False
        # if not all residues the same at position j
        elif self.same_residue and len(set(possible_j_codes)) != 1:
            return False
        # same residue at the same position in both sequence indices. This is a
        # good contact.
        else:
            return True

    def _create_feat_dict(self):
        feat_dict = {}
        pair_dict = {}
        for protein in self.protein_list:
            pair_dict[protein] = []
        for contact in self.contacts:
            # contact is valid if we have the same residue at that position for all sequences in
            # the alignment
            if self._valid_contact(contact):
                seq_ind_i, seq_ind_j = contact
                for protein in self.protein_list:
                    inv_map = self.all_inv_mappings[protein]
                    residue_index_i = inv_map[seq_ind_i]
                    residue_index_j = inv_map[seq_ind_j]
                    pair_dict[protein].append(
                        [residue_index_i, residue_index_j])
        for protein in self.protein_list:
            # create a custom ContactFeaturizer for this particular sequence.
            if self.soft_min:
                feat_dict[protein] = ContactFeaturizer(pair_dict[protein], scheme=self.scheme,
                                                       ignore_nonprotein=self.ignore_nonprotein,
                                                       soft_min=self.soft_min,
                                                       soft_min_beta=self.soft_min_beta)
            else:
                feat_dict[protein] = ContactFeaturizer(pair_dict[protein], scheme=self.scheme,
                                                       ignore_nonprotein=self.ignore_nonprotein)
        return feat_dict

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

        Warning
        -------
        Only works for chain 0 for now.
        """
        seq_id = [k for k, v in self.all_sequences.items()
                  if v == traj.top.to_fasta(chain=0)][0]
        return self.feat_dict[seq_id].partial_transform(traj)

    def _transform(self, distances):
        return distances

    def describe_features(self, traj):
        seq_id = [k for k, v in self.all_sequences.items()
                  if v == traj.top.to_fasta(chain=0)][0]
        return self.feat_dict[seq_id].describe_features(traj)

    def _map_residue_ind_seq_ind(self, alignment):
        all_mappings = {}
        all_sequences = {}
        all_inv_mappings = {}
        for prt in alignment.keys():
            aligned_seq = alignment[prt]
            prt_seq = ''.join([i for i in aligned_seq if i != "-"])

            all_sequences[prt] = prt_seq

            mapping = {}
            seq_index = 0

            for i in range(len(prt_seq)):
                while True:
                    if prt_seq[i] == aligned_seq[seq_index]:
                        mapping[i] = seq_index
                        seq_index += 1
                        break
                    else:
                        seq_index += 1
                        continue
            all_mappings[prt] = mapping
            all_inv_mappings[prt] = {v: k for k, v in mapping.items()}
        return all_inv_mappings, all_sequences, all_mappings
