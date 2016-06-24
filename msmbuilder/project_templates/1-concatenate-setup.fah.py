from msmbuilder.io import load_meta, save_meta
from msmbuilder.utils.fah import continuous_gens, stride_gens

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
stride_info = stride_gens(
    meta,
    want_step_ns=0.960,
    want_traj_ns=960,
    levels_traj=['proj', 'run', 'clone']
)

## Save
save_meta(meta)
save_meta(stride_info, 'stride_info.pandas.pickl')
