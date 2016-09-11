"""Cluster based on RMSD between conformations

{{header}}

Meta
----
depends:
  - meta.pandas.pickl
  - trajs
  - top.pdb
"""
import mdtraj as md

from msmbuilder.cluster import MiniBatchKMedoids
from msmbuilder.io import load_meta, itertrajs, save_trajs, preload_top

## Set up parameters
kmed = MiniBatchKMedoids(
    n_clusters=500,
    metric='rmsd',
)

## Load
meta = load_meta()
centroids = md.load("centroids.xtc", top=preload_top(meta))

## Kernel
SIGMA = 0.3  # nm
from msmbuilder.featurizer import RMSDFeaturizer
import numpy as np

featurizer = RMSDFeaturizer(centroids)
lfeats = {}
for i, traj in itertrajs(meta):
    lfeat = featurizer.partial_transform(traj)
    lfeat = np.exp(-lfeat ** 2 / (2 * (SIGMA ** 2)))
    lfeats[i] = lfeat
save_trajs(lfeats, 'ftrajs', meta)
