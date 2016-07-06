"""Sample conformations from clusters

{{header}}

Meta
----
depends:
  - ../../top.pdb
  - ../../trajs
"""

import mdtraj as md
import os

from msmbuilder.io.sampling import sample_states
from msmbuilder.io import load_trajs, save_generic, preload_top, backup, load_generic

## Load
meta, ttrajs = load_trajs('ttrajs')
kmeans = load_generic("kmeans.pickl")

## Sample
inds = sample_states(ttrajs,
                     kmeans.cluster_centers_,
                     k=10)

save_generic(inds, "cluster-sample-inds.pickl")

## Make trajectories
top = preload_top(meta)
out_folder = "cluster_samples"
backup(out_folder)
os.mkdir(out_folder)

for state_i, state_inds in enumerate(inds):
    traj = md.join(
        md.load_frame(meta.loc[traj_i]['traj_fn'], index=frame_i, top=top)
        for traj_i, frame_i in state_inds
    )
    traj.save("{}/{}.xtc".format(out_folder, state_i))
