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
    trajs : sequence of np.ndarray
        List of tica-transformed trajectories
    dimension : int
        dimension to sample on
    n_frames : int
        Number of frames required
    scheme : {'linear', 'random', 'edges'}
        'linear' samples the tic linearly, 'random' samples randomly
        (thereby taking approximate free energies into account),
        and 'edges' samples the edges of the tic only.

    Returns
    -------
    inds : list of tuples
       Tuples of (trajectory_index, frame_index)
    """
    trajs = [traj[:, [dimension]] for traj in trajs]

    # sort it because all three sampling schemes use it
    all_vals = []
    for traj in trajs:
        all_vals.extend(traj)
    all_vals = np.sort(all_vals)

    if scheme == "linear":
        max_val = all_vals[-1]
        min_val = all_vals[0]
        spaced_points = np.linspace(min_val, max_val, n_frames)
    elif scheme == "random":
        spaced_points = np.sort(np.random.choice(all_vals, n_frames))
    elif scheme == "edge":
        _cut_point = n_frames // 2
        spaced_points = np.hstack((all_vals[:_cut_point],
                                   all_vals[-_cut_point:]))
    else:
        raise ValueError("Scheme has be to one of linear, random or edge")

    tree = KDTree(trajs)
    dists, inds = tree.query(spaced_points)
    return inds


def sample_region(trajs, pt_dict, n_frames, ):
    """Sample a region of the data.

    Parameters
    ----------
    trajs : sequence of np.ndarray
        List of tica-transformed trajectories
    pt_dict : dict
        Dictionary where the keys are the dimensions and the
        value is the value of the dimension.
        E.g., ``pt = {0: 0.15, 4: 0.2}``
    n_frames: int
        Number of frames required

    Returns
    -------
    inds : list of tuples
       Tuples of (trajectory_index, frame_index)
    """
    dimensions = list(pt_dict.keys())
    trajs = [traj[:, dimensions] for traj in trajs]

    tree = KDTree(trajs)
    pt = [pt_dict[i] for i in dimensions]
    dists, inds = tree.query(pt, n_frames)
    return inds
