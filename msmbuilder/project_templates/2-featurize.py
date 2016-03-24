"""Turn trajectories into dihedral features

{{header}}
"""
import mdtraj as md

from msmbuilder.io import load_meta, preload_tops, save_trajs, save_generic
from msmbuilder.featurizer import DihedralFeaturizer

## Load
meta = load_meta()
tops = preload_tops(meta)
dihed_feat = DihedralFeaturizer()


## Logic for lazy-loading trajectories
# TODO: refactor into msmbuilder.io
def trajectories(stride=1):
    for i, row in meta.iterrows():
        yield i, md.load(row['traj_fn'],
                         top=tops[row['top_fn']],
                         stride=stride)


## Featurize
dihed_trajs = {}
for i, traj in trajectories():
    dihed_trajs[i] = dihed_feat.partial_transform(traj)

## Summary
print(dihed_feat.summarize())

## Save
save_trajs(dihed_trajs, 'diheds', meta)
save_generic(dihed_feat, 'diheds.pickl')
