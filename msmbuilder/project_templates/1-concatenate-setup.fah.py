import pandas as pd

from msmbuilder.io import load_meta, save_meta
from msmbuilder.utils.fah import continuous_gens

## Load
meta = load_meta()

## Consider overlapping frames
meta['true_nframes'] = meta['nframes'] - meta['overlapping_frame']

## Make sure we have continuous gens
meta, problematic = continuous_gens(meta, ['proj', 'run', 'clone'], 'gen')
for pr in problematic:
    print("WARNING! Droping gens from {p[index]} after {p[last_good]}"
          .format(p=pr))

## Set up striding
want_step_ns = 0.960
want_traj_ns = 960
new_steps_per_traj = want_traj_ns / want_step_ns
assert int(new_steps_per_traj) == new_steps_per_traj
new_steps_per_traj = int(new_steps_per_traj)
meta['to_stride'] = want_step_ns / (meta['step_ps'] / 1000)


## Only take what we need
def _careful_stride(group):
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


def careful_stride(group):
    return pd.DataFrame(_careful_stride(group), columns=['gen', 'frame'])


## Make a new dataframe
stride_info = meta.groupby(level=['proj', 'run', 'clone']).apply(careful_stride)

## Save
print(stride_info.head())
print(stride_info.tail())
save_meta(meta)
save_meta(stride_info, 'stride_info.pandas.pickl')
