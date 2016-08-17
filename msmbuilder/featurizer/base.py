# Author: Kyle A. Beauchamp <kyleabeauchamp@gmail.com>
# Contributors: Robert McGibbon <rmcgibbo@gmail.com>
# Copyright (c) 2016, Stanford University and the Authors
# All rights reserved.

import mdtraj as md
import numpy as np
from sklearn.base import TransformerMixin

from ..base import BaseEstimator


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
    At the bare minimum, a featurizer must implement the
    `partial_transform(traj)` member function. A `transform(traj_list)`
    for featurizing multiple trajectories in batch will be provided.
    """

    def __init__(self):
        pass

    def featurize(self, traj):
        raise NotImplementedError('This API was removed. '
                                  'Use partial_transform instead')

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
