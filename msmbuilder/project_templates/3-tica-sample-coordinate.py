"""Sample tICA coordinates

{{header}}
"""

import mdtraj as md

from msmbuilder.decomposition.interpretation import sample_dimension
from msmbuilder.io import load_trajs, save_generic, preload_top, backup

## Load
meta, ttrajs = load_trajs('ttrajs')

## Sample
inds = sample_dimension(ttrajs,
                        dimension=0,
                        n_frames=200, scheme='random')

save_generic(inds, "tica-dimension-0-inds.pickl")

## Make trajectory
top = preload_top(meta)

traj = md.join(
    md.load_frame(meta.loc[traj_i]['traj_fn'], index=frame_i, top=top)
    for traj_i, frame_i in inds
)

## Save
traj_fn = "tica-dimension-0.xtc"
backup(traj_fn)
traj.save(traj_fn)
