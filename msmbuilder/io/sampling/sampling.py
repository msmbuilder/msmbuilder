# Author: Muneeb Sultan <msultan@stanford.edu>
# Contributors: Matthew Harrigan <matthew.harrigan@outlook.com>
# Copyright (c) 2016, Stanford University and the Authors
# All rights reserved.

from __future__ import absolute_import
import numpy as np
from ...utils import KDTree


def sample_dimension(trajs, dimension, n_frames, scheme="linear"):
    """Sample a dimension of the data.

    This method uses one of 3 schemes. All other dimensions are ignored, so
    this might result in a really "jumpy" sampled trajectory.

    Parameters
    ----------
    trajs : dictionary of np.ndarray
        Dictionary of tica-transformed trajectories, keyed by arbitrary keys.
        The resulting trajectory indices will use these keys.
    dimension : int
        dimension to sample on
    n_frames : int
        Number of frames requested
    scheme : {'linear', 'random', 'edges'}
        'linear' samples the tic linearly, 'random' samples randomly
        (thereby taking approximate free energies into account),
        and 'edges' samples the edges of the tic only.

    Returns
    -------
    inds : list of tuples
       Tuples of (trajectory_index, frame_index), where trajectory_index is
       in the domain of the keys of the input dictionary.
    """
    fixed_indices = list(trajs.keys())
    trajs = [trajs[k][:, [dimension]] for k in fixed_indices]

    # sort it because all three sampling schemes use it
    all_vals = []
    for traj in trajs:
        all_vals.extend(traj[:, 0])
    all_vals = np.sort(all_vals)

    if scheme == "linear":
        max_val = all_vals[-1]
        min_val = all_vals[0]
        spaced_points = np.linspace(min_val, max_val, n_frames)
    elif scheme == "random":
        spaced_points = np.sort(np.random.choice(all_vals, n_frames))[:,
                        np.newaxis]
    elif scheme == "edge":
        _cut_point = n_frames // 2
        spaced_points = np.hstack((all_vals[:_cut_point],
                                   all_vals[-_cut_point:]))
    else:
        raise ValueError("Scheme has be to one of linear, random or edge")

    tree = KDTree(trajs)
    dists, inds = tree.query(spaced_points)
    return [(fixed_indices[i], j) for i, j in inds]


def sample_states(trajs, state_centers, k=1):
    fixed_indices = list(trajs.keys())
    trajs = [trajs[k] for k in fixed_indices]
    tree = KDTree(trajs)
    dists, inds = tree.query(state_centers, k=k)
    if k > 1:
        # inds; (query_point, k number, (traj_i, frame_i))
        return [
            [(fixed_indices[i], j) for i, j in qinds]
            for qinds in inds
            ]
    else:
        return [(fixed_indices[i], j) for i, j in inds]
