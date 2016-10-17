# Author: Carlos Xavier Hernandez <cxh@stanford.edu>
# Contributors:
# Copyright (c) 2016, Stanford University and the Authors
# All rights reserved.
import numpy as np
from collections import OrderedDict

from ..featurizer import Featurizer


class FeatureSelector(Featurizer):
    """Concatenates results of multiple feature extraction objects.

    This estimator applies a list of feature_extraction objects then
    concatenates the results. This is useful to combine several feature
    extraction mechanisms into a single transformer.

    Note: Users should consider using `msmbuilder.preprocessing.StandardScaler`
    to normalize their data after combining feature sets.

    Parameters
    ----------
    features : list of (str, msmbuilder.feature_extraction) tuples
        List of feature_extraction objects to be applied to the data.
        The first half of each tuple is the name of the feature_extraction.
    which_feat : list or str
        Either a string or a list of strings of features to include in the
        transformer.
    """

    @property
    def which_feat(self):
        return self._which_feat

    @which_feat.setter
    def which_feat(self, value):
        if not isinstance(value, list):
            value = [value]
        elif not all([feat in self.feat_list for feat in value]):
            raise ValueError('Not a valid feature')
        self._which_feat = value

    def __init__(self, features, which_feat=None):
        self.features = OrderedDict(features)
        self.feat_list = list(self.features)

        which_feat = which_feat if which_feat else self.feat_list[:]

        self.which_feat = which_feat

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
        """
        return np.concatenate([self.features[feat].partial_transform(traj)
                               for feat in self.which_feat], axis=1)

    def describe_features(self, traj):
        """ Return a list of dictionaries describing the features. Follows
        the ordering of featurizers in self.which_feat.

        Parameters
        ----------
        traj : mdtraj.Trajectory
            The trajectory to describe

        Returns
        -------
        feature_descs : list of dict
            Dictionary describing each feature with the following information
            about the atoms participating in each feature
                - resnames: unique names of residues
                - atominds: atom indicies involved in the feature
                - resseqs: unique residue sequence ids (not necessarily
                  0-indexed)
                - resids: unique residue ids (0-indexed)
                - featurizer: featurizer dependent
                - featuregroup: other info for the featurizer
        """
        all_res = []
        for feat in self.which_feat:
            all_res.extend(self.features[feat].describe_features(traj))
        return all_res
