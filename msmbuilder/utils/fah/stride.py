import pandas as pd

want_step_ns = 0.960
want_traj_ns = 960


def _careful_stride(group, new_steps_per_traj):
    framei = 0
    geni = 0
    inds = []

    for outi in range(new_steps_per_traj):
        gen = group.index[geni][-1]
        inds += [(gen, framei)]

        framei += group.iloc[geni]['to_stride']
        while framei >= group.iloc[geni]['true_nframes']:
            framei -= group.iloc[geni]['true_nframes']
            geni += 1

            if geni >= len(group):
                return inds

    return inds


def stride_gens(meta, want_step_ns, want_traj_ns,
                levels_traj=None):
    """Stride gens where the stride may be larger than the number of frames

    Parameters
    ----------
    meta : pd.DataFrame
        Dataframe with metadata about trajectory parts (gens).
    want_step_ns : float
        Our desired number of nanoseconds in the strided trajectories
    want_traj_ns : float
        Our desired number of nanoseconds in concatenated trajectories
    levels_traj : list of pd.MultiIndex names.
        Grouping by these multiindex levels should give one continuous
        trajectory per group. If None, defaults to ['proj', 'run', 'clone'].

    Returns
    -------
    stride_info : pd.DataFrame
        DataFrame indexed by ``levels_traj`` with two columns: 'gen' and
        'frame' giving the (gen, frame) index from which you should select
        frames for a strided and concatenated trajectory
    """

    assert "true_nframes" in meta.columns, "We need a column named true_nframes"
    assert "step_ps" in meta.columns, "We need a column named step_ps"

    if levels_traj is None:
        levels_traj = ['proj', 'run', 'clone']

    new_steps_per_traj = want_traj_ns / want_step_ns
    assert int(new_steps_per_traj) == new_steps_per_traj
    new_steps_per_traj = int(new_steps_per_traj)

    meta = meta.copy()
    meta['to_stride'] = want_step_ns / (meta['step_ps'] / 1000)

    def careful_stride(group):
        """Wrap numpy array into a dataframe"""
        return pd.DataFrame(_careful_stride(group, new_steps_per_traj),
                            columns=['gen', 'frame'])

    stride_info = meta.groupby(level=levels_traj).apply(careful_stride)
    return stride_info
