from __future__ import print_function, division, absolute_import
import numpy as np
import mdtraj as md

__all__ = ['map_drawn_samples']


def map_drawn_samples(selected_pairs_by_state, trajectories, top=None):
    """Lookup trajectory frames using pairs of (trajectory, frame) indices.

    Parameters
    ----------
    selected_pairs_by_state : array, dtype=int, shape=(n_states, n_samples, 2)
        selected_pairs_by_state[state, sample] gives the (trajectory, frame)
        index associated with a particular sample from that state.
    trajectories : list(md.Trajectory) or list(np.ndarray) or list(filenames)
        The trajectories assocated with sequences,
        which will be used to extract coordinates of the state centers
        from the raw trajectory data.  This can also be a list of np.ndarray
        objects or filenames.  If they are filenames, mdtraj will be used to
        load them
    top : md.Topology, optional, default=None
        Use this topology object to help mdtraj load filenames

    Returns
    -------
    frames_by_state : mdtraj.Trajectory
        Output will be a list of trajectories such that frames_by_state[state]
        is a trajectory drawn from `state` of length `n_samples`.  If
        trajectories are numpy arrays, the output will be numpy arrays instead
        of md.Trajectories

    Examples
    --------
    >>> selected_pairs_by_state = hmm.draw_samples(sequences, 3)
    >>> samples = map_drawn_samples(selected_pairs_by_state, trajectories)

    Notes
    -----
    YOU are responsible for ensuring that selected_pairs_by_state and
    trajectories correspond to the same dataset!

    See Also
    --------
    ghmm.GaussianHMM.draw_samples : Draw samples from GHMM
    ghmm.GaussianHMM.draw_centroids : Draw centroids from GHMM
    """

    frames_by_state = []

    for state, pairs in enumerate(selected_pairs_by_state):
        if isinstance(trajectories[0], str):
            if top:
                process = lambda x, frame: md.load_frame(x, frame, top=top)
            else:
                process = lambda x, frame: md.load_frame(x, frame)
        else:
            process = lambda x, frame: x[frame]

        frames = [process(trajectories[trj], frame) for trj, frame in pairs]
        try:  # If frames are mdtraj Trajectories
            # Get an empty trajectory with correct shape and call the join
            # method on it to merge trajectories
            state_trj = frames[0][0:0].join(frames)
        except AttributeError:
            state_trj = np.array(frames)  # Just a bunch of np arrays
        frames_by_state.append(state_trj)

    return frames_by_state
