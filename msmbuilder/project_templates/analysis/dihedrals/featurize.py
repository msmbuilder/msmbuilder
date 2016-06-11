"""Turn trajectories into dihedral features

{{header}}

Meta
----
depends:
  - meta.pandas.pickl
  - trajs
  - top.pdb
"""
import mdtraj as md

from msmbuilder.featurizer import DihedralFeaturizer
from msmbuilder.io import load_meta, preload_tops, save_trajs, save_generic
from multiprocessing import Pool

## Load
meta = load_meta()
tops = preload_tops(meta)
dihed_feat = DihedralFeaturizer()


## Featurize logic
def feat(irow):
    i, row = irow
    traj = md.load(row['traj_fn'], top=tops[row['top_fn']])
    feat_traj = dihed_feat.partial_transform(traj)
    return i, feat_traj


## Do it in parallel
with Pool() as pool:
    dihed_trajs = dict(pool.imap_unordered(feat, meta.iterrows()))

## Save
save_trajs(dihed_trajs, 'ftrajs', meta)
save_generic(dihed_feat, 'featurizer.pickl')
