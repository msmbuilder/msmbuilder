# Author: Matthew Harrigan <matthew.harrigan@outlook.com>
# Contributors:
# Copyright (c) 2015, Stanford University and the Authors
# All rights reserved.

from ..base import BaseEstimator
from sklearn.base import TransformerMixin
import numpy as np
import warnings


class FeatureUnion(BaseEstimator, TransformerMixin):
    def __init__(self, normalize):
        warnings.warn("msmbuilder.featurizer.FeatureUnion is deprecated. "
                      "Please see "
                      "msmbuilder.feature_selection.FeatureSelector",
                      DeprecationWarning)
        self.normalize = normalize
        self._traj_lens = None

    def partial_transform(self, traj_zip):
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
        return np.concatenate([self._dim_match(traj) / norm
                               for traj, norm in zip(traj_zip, self._norms)],
                              axis=1)

    def _check_same_length(self, trajs_tuple):
        """Check that the datasets are the same length"""
        lens = [len(trajs) for trajs in trajs_tuple]
        if len(set(lens)) > 1:
            err = "Each dataset must be the same length. You gave: {}"
            err = err.format(lens)
            raise ValueError(err)

    def _dim_match(self, arr):
        if arr.ndim == 1:
            return arr[:, np.newaxis]
        return arr

    def fit(self, trajs_tuple, y=None):
        self._check_same_length(trajs_tuple)

        if not self.normalize:
            self._norms = np.ones(len(trajs_tuple))
            return self

        norms = []
        for trajs in trajs_tuple:
            traj_all = np.concatenate([t for t in trajs])
            norms += [np.std(traj_all)]
        self._norms = np.asarray(norms)
        return self

    def transform(self, trajs_tuple, y=None):
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
        return [self.partial_transform(traj_zip)
                for traj_zip in zip(*trajs_tuple)]
