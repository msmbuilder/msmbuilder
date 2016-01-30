import mdtraj as md
import numpy as np
from mdtraj.testing import get_fn, eq

from msmbuilder.featurizer import SASAFeaturizer


def _test_sasa_featurizer(t, value):
    sasa = md.shrake_rupley(t)
    rids = np.array([a.residue.index for a in t.top.atoms])

    for i, rid in enumerate(np.unique(rids)):
        mask = (rids == rid)
        eq(value[:, i], np.sum(sasa[:, mask], axis=1))


def test_sasa_featurizer_1():
    t = md.load(get_fn('frame0.h5'))
    value = SASAFeaturizer(mode='residue').partial_transform(t)
    assert value.shape == (t.n_frames, t.n_residues)
    _test_sasa_featurizer(t, value)


def test_sasa_featurizer_2():
    t = md.load(get_fn('frame0.h5'))

    # scramle the order of the atoms, and which residue each is a
    # member of
    df, bonds = t.top.to_dataframe()
    df['resSeq'] = np.random.randint(5, size=(t.n_atoms))
    df['resName'] = df['resSeq']
    t.top = md.Topology.from_dataframe(df, bonds)

    value = SASAFeaturizer(mode='residue').partial_transform(t)
    _test_sasa_featurizer(t, value)
