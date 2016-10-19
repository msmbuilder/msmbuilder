# Author: Muneeb Sultan <msultan@stanford.edu>
# Copyright (c) 2016, Stanford University and the Authors
# All rights reserved.
from __future__ import print_function, division, absolute_import
import numpy as np
import itertools
from .featurizer import Featurizer, ContactFeaturizer
import warnings

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
"""

    def __init__(self, alignment=None, contacts='all',
                 same_residue='True',
                 scheme='closest-heavy', ignore_nonprotein=True):

        if alignment is None:
            raise ValueError("Common contacts requires an\
                              alignment(either list of lists or dict)")
        if type(alignment)==list:
            self.alignment={}
            for k,v in enumerate(alignment):
                self.alignment[k] = v
        elif type(alignment)==dict:
            self.alignment = alignment
        else:
            raise ValueError("Alignment is not of type list or dict.")

        self.protein_list = list(self.alignment.keys())
        self.same_residue = same_residue
        self.wanted_positions = contacts
        if contacts is 'all':
            #use the max length(probably a horrible idea)
            self.wanted_positions = range(max([len(self.alignment[i])
                                               for i in self.alignment.keys()]))
            warnings.warn("All valid pair-wise contacts are being calculated")
        
        self.scheme = scheme
        self.ignore_nonprotein = ignore_nonprotein


        self.all_inv_mappings, \
        self.all_sequences = self._map_residue_ind_seq_ind(self.alignment)

        self.feat_dict = self._create_feat_dict()


    def _create_feat_dict(self):
        feat_dict = {}
        for protein in self.protein_list:
            can_keep=[]
            inv_map = self.all_inv_mappings[protein]

            for position in self.wanted_positions:
                possible_codes = set([self.alignment[p][position] for p in
                                      self.alignment.keys()])
                #ignore all additions
                if not "-" in possible_codes:
                    #if we want the same residue ignore it
                    if self.same_residue and len(set(possible_codes))!=1:
                        continue
                    # get the inverse mapping and add it to the list of can keep
                    residue_index = inv_map[position]
                    can_keep.append(residue_index)

            #sort it because i dont want random bs issues.
            can_keep = np.sort(can_keep)
            #get its pairs
            pairs = [i for i in itertools.combinations(can_keep, 2)]
            #create a custom ContactFeaturizer for this particular sequence.
            feat_dict[protein] = ContactFeaturizer(pairs,scheme=self.scheme,
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
        seq_id = [k for k,v in self.all_sequences.items()
                  if v == traj.top.to_fasta(chain=0)][0]
        return self.feat_dict[seq_id].partial_transform(traj)

    def _transform(self, distances):
        return distances

    def describe_features(self, traj):
        seq_id = [k for k,v in self.all_sequences.items()
                  if v == traj.top.to_fasta(chain=0)][0]
        return self.feat_dict[seq_id].describe_features(traj)

    def _map_residue_ind_seq_ind(self,alignment):
        all_mappings = {}
        all_sequences = {}
        all_inv_mappings = {}
        for prt in alignment.keys():
            aligned_seq = alignment[prt]
            prt_seq = ''.join([i for i in aligned_seq if i!="-"])

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
        return all_inv_mappings, all_sequences

