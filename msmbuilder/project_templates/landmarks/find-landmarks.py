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
from msmbuilder.io import load_meta, itertrajs, save_generic, backup

## Set up parameters
kmed = MiniBatchKMedoids(
    n_clusters=500,
    metric='rmsd',
)

## Load
meta = load_meta()


## Try to limit RAM usage
def guestimate_stride():
    total_data = meta['nframes'].sum()
    want = kmed.n_clusters * 10
    stride = max(1, total_data // want)
    print("Since we have", total_data, "frames, we're going to stride by",
          stride, "during fitting, because this is probably adequate for",
          kmed.n_clusters, "clusters")
    return stride


## Fit
kmed.fit([traj for _, traj in itertrajs(meta, stride=guestimate_stride())])
print(kmed.summarize())

## Save
save_generic(kmed, 'clusterer.pickl')


## Save centroids
def frame(traj_i, frame_i):
    # Note: kmedoids does 0-based, contiguous integers so we use .iloc
    row = meta.iloc[traj_i]
    return md.load_frame(row['traj_fn'], frame_i, top=row['top_fn'])


centroids = md.join((frame(ti, fi) for ti, fi in kmed.cluster_ids_),
                    check_topology=False)
centroids_fn = 'centroids.xtc'
backup(centroids_fn)
centroids.save("centroids.xtc")
