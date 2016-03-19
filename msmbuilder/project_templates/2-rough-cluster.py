import mdtraj as md
import pandas as pd

from msmbuilder.cluster import MiniBatchKMedoids
from msmbuilder.utils import dump
from msmbuilder.dataset2 import save

# Set parameters here
kmed = MiniBatchKMedoids(
    n_clusters=500,
    metric='rmsd',
)

# Load
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


def guestimate_stride():
    total_data = meta['nframes'].sum()
    want = kmed.n_clusters * 10
    stride = total_data // want
    print("Since we have", total_data, "frames, we're going to stride by",
          stride, "during fitting, because this is probably adequate for",
          kmed.n_clusters, "clusters")
    return stride


# Fit
kmed.fit([traj for _, traj in trajectories(stride=guestimate_stride())])

# Transform
ktrajs = {}
for i, traj in trajectories():
    ktrajs[i] = kmed.partial_transform(traj)

print(kmed.summarize())

# Save
save(meta, ktrajs, 'rmsd-ktrajs')
dump(kmed, 'rmsd-kmedoids.pickl')
