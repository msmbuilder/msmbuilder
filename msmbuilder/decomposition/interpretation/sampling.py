# Author: Muneeb Sultan <msultan@stanford.edu>
# Copyright (c) 2016, Stanford University and the Authors
# All rights reserved.

from __future__ import absolute_import
import numpy as np
from ...utils import KDTree


def sample_dimension(data, dimension, n_frames, scheme="linear"):
    """Function to sample a dimension of the data
    using one of 3 schemes. All other dimensions are ignored.

    Parameters
    ----------
    data : list of lists
        List of low dimensional data(output of tica)
    dimension : int
        dimension to sample on
    n_frames: int
        Number of frames required
    scheme: string
        One of either "linear", "random" or "edges". Linear
        samples the tic linearly, random samples randomly
        thereby taking approximate free energies into account,
        and edges samples the edges of the tic only.

    Returns
    -------
       list of tuples where first number is the trajectory index and
       second is the frame index
    """
    d_data = [i[:,dimension][:,np.newaxis] for i in data]

    #sort it because all three sampling schemes use it

    all_vals = []
    for i in d_data:
        all_vals.extend(i)
    all_vals = np.sort(all_vals)

    #get lineraly placed points
    if scheme=="linear":
        max_val = all_vals[-1]
        min_val = all_vals[0]
        spaced_points = np.linspace(min_val, max_val, n_frames)

    elif scheme=="random":
        spaced_points = np.sort(np.random.choice(all_vals, n_frames))

    elif scheme=="edge":
        _cut_point = np.int(n_frames / 2)
        spaced_points = np.hstack((all_vals[:_cut_point], all_vals[-_cut_point:]))
    else:
        raise ValueError("Scheme has be to one of linear, random or edge")

    tree = KDTree(d_data)

    return_vec = []
    for pt in spaced_points:
        dis, ind = tree.query([pt])
        return_vec.append(ind)

    return return_vec


def sample_region(data, pt_dict, n_frames,):
    """Function to sample a region of the data.

    Parameters
    ----------
    data : list of lists
        List of low dimensional data(output of tica)
    pt_dict : dict
        Dictionary where the keys are the dimensions and the
        value is the value of the dimension.
        pt = {0:0.15, 4:0.2}
    n_frames: int
        Number of frames required

    Returns
    -------
       list of tuples where first number is the trajectory index and
       second is the frame index
    """
    dimensions = list(pt_dict.keys())
    d_data = [i[:, dimensions] for i in data]

    tree = KDTree(d_data)
    pt = [pt_dict[i] for i in dimensions]
    dis, ind = tree.query(pt, n_frames)
    return ind
