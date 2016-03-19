import mdtraj as md
import pandas as pd

from msmbuilder.dataset2 import save
from msmbuilder.featurizer import VonMisesFeaturizer
from msmbuilder.utils import dump

vmfeat = VonMisesFeaturizer()

meta = pd.read_pickle("meta.pandas.pickl")


def preload_tops():
    top_fns = set(meta['top_fn'])
    tops = {}
    for tfn in top_fns:
        tops[tfn] = md.load(tfn)
    return tops


tops = preload_tops()


def trajectories(stride=1):
    for i, row in meta.iterrows():
        yield i, md.load(row['traj_fn'],
                         top=tops[row['top_fn']],
                         stride=stride)


vmtrajs = {}
for i, traj in trajectories():
    vmtrajs[i] = vmfeat.partial_transform(traj)

print(vmfeat.summarize())

# Save
save(meta, vmtrajs, 'vmtrajs')
dump(vmfeat, 'vmfeat.pickl')
