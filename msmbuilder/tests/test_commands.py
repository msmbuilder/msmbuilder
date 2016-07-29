from __future__ import print_function, division

import glob
import itertools
import json
import os
import shlex
import shutil
import subprocess
import tempfile

import hmmlearn.hmm
import mdtraj as md
import numpy as np
from mdtraj.testing import eq
from mdtraj.testing import get_fn as get_mdtraj_fn

from msmbuilder.dataset import dataset
from msmbuilder.example_datasets import get_data_home
from msmbuilder.utils import load

DATADIR = HMM = None


def setup_module():
    global DATADIR, HMM
    DATADIR = tempfile.mkdtemp()
    # 4 components and 3 features. Each feature is going to be the x, y, z
    # coordinate of 1 atom
    HMM = hmmlearn.hmm.GaussianHMM(n_components=4)
    HMM.transmat_ = np.array([[0.9, 0.1, 0.0, 0.0],
                              [0.1, 0.7, 0.2, 0.0],
                              [0.0, 0.1, 0.8, 0.1],
                              [0.0, 0.1, 0.1, 0.8]])
    HMM.means_ = np.array([[-10, -10, -10],
                           [-5, -5, -5],
                           [5, 5, 5],
                           [10, 10, 10]])
    HMM.covars_ = np.array([[0.1, 0.1, 0.1],
                            [0.5, 0.5, 0.5],
                            [1, 1, 1],
                            [4, 4, 4]])
    HMM.startprob_ = np.array([1, 1, 1, 1]) / 4.0

    # get a 1 atom topology
    topology = md.load(get_mdtraj_fn('native.pdb')).atom_slice([1]).topology

    # generate the trajectories and save them to disk
    for i in range(10):
        d, s = HMM.sample(100)
        t = md.Trajectory(xyz=d.reshape(len(d), 1, 3), topology=topology)
        t.save(os.path.join(DATADIR, 'Trajectory%d.h5' % i))

    assert os.path.exists("{}/alanine_dipeptide".format(get_data_home()))


def teardown_module():
    shutil.rmtree(DATADIR)


class tempdir(object):
    def __enter__(self):
        self._curdir = os.path.abspath(os.curdir)
        self._tempdir = tempfile.mkdtemp()
        os.chdir(self._tempdir)

    def __exit__(self, *exc_info):
        os.chdir(self._curdir)
        shutil.rmtree(self._tempdir)


def shell(str):
    split = shlex.split(str, posix=False)
    try:
        subprocess.check_output(split, stderr=subprocess.STDOUT,
                                universal_newlines=True)
    except subprocess.CalledProcessError as e:
        print(e.cmd)
        print(e.output)
        raise


def test_atomindices_1():
    fn = get_mdtraj_fn('2EQQ.pdb')
    t = md.load(fn)
    with tempdir():
        shell('msmb AtomIndices -o all.txt --all -a -p %s' % fn)
        shell('msmb AtomIndices -o all-pairs.txt --all -d -p %s' % fn)
        atoms = np.loadtxt('all.txt', int)
        pairs = np.loadtxt('all-pairs.txt', int)
        eq(t.n_atoms, len(atoms))
        eq(int(t.n_atoms * (t.n_atoms - 1) / 2), len(pairs))


def test_atomindices_2():
    fn = get_mdtraj_fn('2EQQ.pdb')
    t = md.load(fn)
    with tempdir():
        shell('msmb AtomIndices -o heavy.txt --heavy -a -p %s' % fn)
        shell('msmb AtomIndices -o heavy-pairs.txt --heavy -d -p %s' % fn)
        atoms = np.loadtxt('heavy.txt', int)
        pairs = np.loadtxt('heavy-pairs.txt', int)
        assert all(t.topology.atom(i).element.symbol != 'H' for i in atoms)
        assert (sum(1 for a in t.topology.atoms if a.element.symbol != 'H') ==
                len(atoms))
        eq(np.array(list(itertools.combinations(atoms, 2))), pairs)


def test_atomindices_3():
    fn = get_mdtraj_fn('2EQQ.pdb')
    t = md.load(fn)
    with tempdir():
        shell('msmb AtomIndices -o alpha.txt --alpha -a -p %s' % fn)
        shell('msmb AtomIndices -o alpha-pairs.txt --alpha -d -p %s' % fn)
        atoms = np.loadtxt('alpha.txt', int)
        pairs = np.loadtxt('alpha-pairs.txt', int)
        assert all(t.topology.atom(i).name == 'CA' for i in atoms)
        assert sum(1 for a in t.topology.atoms if a.name == 'CA') == len(atoms)
        eq(np.array(list(itertools.combinations(atoms, 2))), pairs)


def test_atomindices_4():
    fn = get_mdtraj_fn('2EQQ.pdb')
    t = md.load(fn)
    with tempdir():
        shell('msmb AtomIndices -o minimal.txt --minimal -a -p %s' % fn)
        shell('msmb AtomIndices -o minimal-pairs.txt --minimal -d -p %s' % fn)
        atoms = np.loadtxt('minimal.txt', int)
        pairs = np.loadtxt('minimal-pairs.txt', int)
        assert all(t.topology.atom(i).name in ['CA', 'CB', 'C', 'N', 'O']
                   for i in atoms)
        eq(np.array(list(itertools.combinations(atoms, 2))), pairs)


def test_superpose_featurizer():
    with tempdir():
        shell('msmb AtomIndices -o all.txt --all -a -p '
              '%s/alanine_dipeptide/ala2.pdb' % get_data_home()),
        shell("msmb SuperposeFeaturizer "
              "--trjs '{data_home}/alanine_dipeptide/*.dcd'"
              " --transformed distances --atom_indices all.txt"
              " --reference_traj {data_home}/alanine_dipeptide/ala2.pdb"
              " --top {data_home}/alanine_dipeptide/ala2.pdb"
              .format(data_home=get_data_home()))
        ds = dataset('distances')
        assert len(ds) == 10
        assert ds[0].shape[1] == len(np.loadtxt('all.txt'))
        print(ds.provenance)


def test_superpose_featurizer_reftop():
    # see issue #555
    with tempdir():
        shell('msmb AtomIndices -o all.txt --all -a -p '
              '%s/alanine_dipeptide/ala2.pdb' % get_data_home()),
        shell("msmb SuperposeFeaturizer "
              "--trjs '{data_home}/alanine_dipeptide/*.dcd'"
              " --transformed distances --atom_indices all.txt"
              " --reference_traj {data_home}/alanine_dipeptide/trajectory-0.dcd"
              " --top {data_home}/alanine_dipeptide/ala2.pdb"
              .format(data_home=get_data_home()))
        ds = dataset('distances')
        assert len(ds) == 10
        assert ds[0].shape[1] == len(np.loadtxt('all.txt'))
        print(ds.provenance)


def test_atom_pairs_featurizer():
    with tempdir():
        shell('msmb AtomIndices -o all.txt --all -d -p '
              '%s/alanine_dipeptide/ala2.pdb' % get_data_home()),
        shell("msmb AtomPairsFeaturizer "
              "--trjs '{data_home}/alanine_dipeptide/*.dcd'"
              " --transformed pairs --pair_indices all.txt"
              " --top {data_home}/alanine_dipeptide/ala2.pdb"
              .format(data_home=get_data_home()))
        ds = dataset('pairs')
        assert len(ds) == 10
        assert ds[0].shape[1] == len(np.loadtxt('all.txt') ** 2)
        print(ds.provenance)


def test_transform_command_1():
    with tempdir():
        shell("msmb KCenters -i {data_home}/alanine_dipeptide/*.dcd "
              "-o model.pkl --top {data_home}/alanine_dipeptide/ala2.pdb "
              "--metric rmsd".format(data_home=get_data_home()))
        shell("msmb TransformDataset -i {data_home}/alanine_dipeptide/*.dcd "
              "-m model.pkl -t transformed.h5 --top "
              "{data_home}/alanine_dipeptide/ala2.pdb"
              .format(data_home=get_data_home()))

        eq(dataset('transformed.h5')[0], load('model.pkl').labels_[0])

    with tempdir():
        shell("msmb KCenters -i {data_home}/alanine_dipeptide/trajectory-0.dcd "
              "-o model.pkl --top {data_home}/alanine_dipeptide/ala2.pdb "
              "--metric rmsd".format(data_home=get_data_home()))


def test_transform_command_2():
    with tempdir():
        shell("msmb KCenters -i {data_home}/alanine_dipeptide/trajectory-0.dcd "
              "-o model.pkl --top {data_home}/alanine_dipeptide/ala2.pdb "
              "--metric rmsd "
              "--stride 2".format(data_home=get_data_home()))


def test_help():
    shell('msmb -h')


def test_convert_chunked_project_1():
    with tempdir():
        root = os.path.join(get_data_home(), 'alanine_dipeptide')
        assert os.path.exists(root)
        cmd = ("msmb ConvertChunkedProject out {root} --pattern *.dcd "
               "-t {root}/ala2.pdb".format(root=root))
        shell(cmd)
        assert set(os.listdir('out')) == {'traj-00000000.dcd',
                                          'trajectories.jsonl'}

        # check that out/traj-00000.dcd really has concatenated all of
        # the input trajs
        length = len(md.open('out/traj-00000000.dcd'))
        assert length == sum(len(md.open(f))
                             for f in glob.glob('%s/*.dcd' % root))

        with open('out/trajectories.jsonl') as f:
            record = json.load(f)
        assert set(record.keys()) == {'filename', 'chunks'}
        assert record['filename'] == 'traj-00000000.dcd'
        assert sorted(glob.glob('%s/*.dcd' % root)) == record['chunks']
