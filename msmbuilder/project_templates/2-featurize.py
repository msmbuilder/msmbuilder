import mdtraj as md

from msmbuilder.dataset2 import (load_meta, preload_tops,
                                 save_trajs, save_generic)
from msmbuilder.featurizer import DihedralFeaturizer

meta = load_meta()
tops = preload_tops(meta)
dihed_feat = DihedralFeaturizer()


# TODO: refactor into dataset2
def trajectories(stride=1):
    for i, row in meta.iterrows():
        yield i, md.load(row['traj_fn'],
                         top=tops[row['top_fn']],
                         stride=stride)


dihed_trajs = {}
for i, traj in trajectories():
    dihed_trajs[i] = dihed_feat.partial_transform(traj)

print(dihed_feat.summarize())

# Save
save_trajs(dihed_trajs, 'diheds', meta)
save_generic(dihed_feat, 'diheds.pickl')
