import mdtraj as md
import pandas as pd

from msmbuilder.dataset2 import save
from msmbuilder.featurizer import DihedralFeaturizer
from msmbuilder.utils import dump

dihed_feat = DihedralFeaturizer()

meta = pd.read_pickle("meta.pandas.pickl")


# TODO: refactor into dataset2
def preload_tops():
    top_fns = set(meta['top_fn'])
    tops = {}
    for tfn in top_fns:
        tops[tfn] = md.load(tfn)
    return tops


tops = preload_tops()


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
save(meta, dihed_trajs, 'diheds')
dump(dihed_feat, 'diheds.pickl')
