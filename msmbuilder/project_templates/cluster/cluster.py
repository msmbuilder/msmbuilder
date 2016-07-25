"""Cluster tICA results

{{header}}

Meta
----
depends:
 - ttrajs
 - meta.pandas.pickl
"""
from msmbuilder.io import load_trajs, save_trajs, save_generic
from msmbuilder.cluster import MiniBatchKMeans

## Load
meta, ttrajs = load_trajs('ttrajs')

## Fit
dim = 5
kmeans = MiniBatchKMeans(n_clusters=500)
kmeans.fit([traj[:, :dim] for traj in ttrajs.values()])

## Transform
ktrajs = {}
for k, v in ttrajs.items():
    ktrajs[k] = kmeans.partial_transform(v[:, :dim])

## Save
print(kmeans.summarize())
save_trajs(ktrajs, 'ktrajs', meta)
save_generic(kmeans, 'kmeans.pickl')
