"""Sample a trajectory from microstate MSM

{{header}}

Meta
----
depends:
  - top.pdb
  - trajs
"""

import mdtraj as md

from msmbuilder.io import load_trajs, save_generic, preload_top, backup, load_generic
from msmbuilder.io.sampling import sample_msm

## Load
meta, ttrajs = load_trajs('ttrajs')
msm = load_generic('msm.pickl')
kmeans = load_generic('kmeans.pickl')

## Sample
# Warning: make sure ttrajs and kmeans centers have
# the same number of dimensions
inds = sample_msm(ttrajs, kmeans.cluster_centers_, msm, n_steps=200, stride=1)
save_generic(inds, "msm-traj-inds.pickl")

## Make trajectory
top = preload_top(meta)
traj = md.join(
    md.load_frame(meta.loc[traj_i]['traj_fn'], index=frame_i, top=top)
    for traj_i, frame_i in inds
)

## Save
traj_fn = "msm-traj.xtc"
backup(traj_fn)
traj.save(traj_fn)
