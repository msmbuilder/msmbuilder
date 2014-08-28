import numpy as np
import mixtape.cluster
import mdtraj as md
import mdtraj.testing

X = 0.3 * np.random.RandomState(0).randn(1000, 10)
trj = md.load(md.testing.get_fn("traj.h5"))

def test_regular_spatial_rmsd():
    model = mixtape.cluster.RegularSpatial(d_min=0.8, metric=md.rmsd)
    model.fit([trj])

def test_regular_spatial():
    model = mixtape.cluster.RegularSpatial(d_min=0.8)
    model.fit([X])

def test_kcenters_rmsd():
    model = mixtape.cluster.KCenters(3, metric=md.rmsd)
    model.fit([trj])

def test_kcenters_spatial():
    model = mixtape.cluster.KCenters(3)
    model.fit([X])
