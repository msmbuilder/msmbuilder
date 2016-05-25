"""Cluster based on RMSD between conformations

{{header}}
"""
import mdtraj as md

from msmbuilder.cluster import MiniBatchKMedoids
from msmbuilder.io import load_meta, preload_tops, save_trajs, save_generic

## Set up parameters
kmed = MiniBatchKMedoids(
    n_clusters=500,
    metric='rmsd',
)

## Load
meta = load_meta()
tops = preload_tops(meta)


## Lazy-load trajectories
def trajectories(stride=1):
    for i, row in meta.iterrows():
        yield i, md.load(row['traj_fn'],
                         top=tops[row['top_fn']],
                         stride=stride)


## Try to limit RAM usage
def guestimate_stride():
    total_data = meta['nframes'].sum()
    want = kmed.n_clusters * 10
    stride = total_data // want
    print("Since we have", total_data, "frames, we're going to stride by",
          stride, "during fitting, because this is probably adequate for",
          kmed.n_clusters, "clusters")
    return stride


## Fit
kmed.fit([traj for _, traj in trajectories(stride=guestimate_stride())])

## Transform
ktrajs = {}
for i, traj in trajectories():
    ktrajs[i] = kmed.partial_transform(traj)

print(kmed.summarize())

## Save
save_trajs(ktrajs, 'rmsd-ktrajs', meta)
save_generic(kmed, 'rmsd-kmedoids.pickl')

## Save centroids
from functools import reduce
from operator import add


def frame(traj_i, frame_i):
    # Note: kmedoids does 0-based, contiguous integers so we use .iloc
    row = meta.iloc[traj_i]
    return md.load_frame(row['traj_fn'], frame_i, top=row['top_fn'])


centroids = reduce(add, (frame(ti, fi) for ti, fi in kmed.cluster_ids_))
centroids.save("rmsd-centroids.xtc")

## Kernel
SIGMA = 0.5  # nm
from msmbuilder.featurizer import RMSDFeaturizer
import numpy as np

featurizer = RMSDFeaturizer(centroids)
lfeats = {}
for i, traj in trajectories():
    lfeat = featurizer.partial_transform(traj)
    lfeat = np.exp(lfeat ** 2 / (2 * (SIGMA ** 2)))
    lfeats[i] = lfeat
save_trajs(lfeats, 'lfeats', meta)
