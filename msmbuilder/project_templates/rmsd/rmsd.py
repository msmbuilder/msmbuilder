"""Check for abnormally high rmsd values to a reference structure

{{header}}

Meta
----
depends:
  - meta.pandas.pickl
  - trajs
  - top.pdb

"""

import mdtraj as md

from msmbuilder.io import load_meta, itertrajs, save_trajs

## Load reference structure
ref = md.load("top.pdb")
meta = load_meta()

## Do calculation and save
rmsds = {k: md.rmsd(traj, ref) for k, traj in itertrajs(meta)}
save_trajs(rmsds, 'rmsds', meta)
