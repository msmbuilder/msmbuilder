import warnings

import mdtraj as md
import numpy as np

from msmbuilder.featurizer import LigandContactFeaturizer
from msmbuilder.featurizer import BinaryLigandContactFeaturizer
from msmbuilder.featurizer import LigandRMSDFeaturizer


def _random_trajs():
    top = md.Topology()
    c = top.add_chain()
    r = top.add_residue('HET', c)
    r2 = top.add_residue('HET', c)
    r3 = top.add_residue('HET', c)
    cx = top.add_chain()
    rx = top.add_residue('HET', cx)
    for _ in range(10):
        top.add_atom('CA', md.element.carbon, r)
        top.add_atom('CA', md.element.carbon, r2)
        top.add_atom('CA', md.element.carbon, r3)
    for _ in range(10):
        top.add_atom('CA', md.element.carbon, rx)
    traj = md.Trajectory(xyz=np.random.uniform(size=(100, 40, 3)),
                          topology=top,
                          time=np.arange(100))
    ref = md.Trajectory(xyz=np.random.uniform(size=(1, 40, 3)),
                        topology=top,
                        time=np.arange(1))
    return traj, ref


def test_chain_guessing():
    traj, ref = _random_trajs()
    feat = LigandContactFeaturizer(reference_frame=ref)
    contacts = feat.transform(traj)

    assert feat.protein_chain == 0
    assert feat.ligand_chain == 1
    assert len(contacts) == 100
    assert contacts[0].shape[1] == 3


def test_binding_pocket():
    traj, ref = _random_trajs()
    feat = LigandContactFeaturizer(reference_frame=ref)
    pocket_ref = feat.transform([ref])
    limit = (max(pocket_ref[0][0]) + min(pocket_ref[0][0]))/2.0
    number_included = sum(pocket_ref[0][0] < limit)


    pocket_feat = LigandContactFeaturizer(reference_frame=ref,
                                          binding_pocket=limit)
    pocket_contacts = pocket_feat.transform(traj)

    assert len(pocket_contacts[0][0]) == number_included


def test_binaries():
    traj, ref = _random_trajs()
    feat = BinaryLigandContactFeaturizer(reference_frame=ref, cutoff=0.1)
    binaries = feat.transform(traj)

    assert np.sum(binaries[:]) <= len(binaries)*binaries[0].shape[1]


def test_binaries_binding_pocket():
    traj, ref = _random_trajs()
    feat = LigandContactFeaturizer(reference_frame=ref)
    pocket_ref = feat.transform([ref])
    limit = (max(pocket_ref[0][0]) + min(pocket_ref[0][0]))/2.0
    cutoff = limit*0.8
    number_included = sum(pocket_ref[0][0] < limit)

    pocket_feat = BinaryLigandContactFeaturizer(reference_frame=ref,
                                                cutoff=cutoff,
                                                binding_pocket=limit)
    pocket_binaries = pocket_feat.transform(traj)

    assert len(pocket_binaries[0][0]) == number_included
    assert (np.sum(pocket_binaries[:]) <=
            len(pocket_binaries)*pocket_binaries[0].shape[1])


def test_single_index_rmsd():
    traj, ref = _random_trajs()
    feat = LigandRMSDFeaturizer(reference_frame=ref,
                                calculate_indices=[ref.n_atoms-1])
    single_cindex = feat.transform([traj])
    assert np.unique(single_cindex).shape[0] > 1
    # this actually won't pass for standard mdtraj rmsd 
    # with len(atom_indices)=1, I think because of the superposition
    # built into the calculation


def test_mdtraj_equivalence():
    traj, ref = _random_trajs()
    feat = LigandRMSDFeaturizer(reference_frame=ref, align_by='custom',
                    calculate_for='custom', align_indices=range(ref.n_atoms),
                     calculate_indices=range(ref.n_atoms))
    multi_chain = feat.transform([traj])
    md_traj = md.rmsd(traj,ref,frame=0)
    np.testing.assert_almost_equal(multi_chain[0][:, 0], md_traj, decimal=4)
