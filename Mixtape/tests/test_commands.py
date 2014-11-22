from __future__ import print_function, division
import os
import itertools
import tempfile
import shutil
import shlex
import subprocess
import numpy as np
import mdtraj as md
from mdtraj.testing import eq
from mdtraj.testing import get_fn as get_mdtraj_fn

from mixtape.dataset import dataset
from mixtape.example_datasets import get_data_home
from mixtape.example_datasets.alanine_dipeptide import fetch_alanine_dipeptide

import sklearn.hmm
DATADIR = HMM = None

################################################################################
# Fixtures
################################################################################

def setup_module():
    global DATADIR, HMM
    DATADIR = tempfile.mkdtemp()
    # 4 components and 3 features. Each feature is going to be the x, y, z
    # coordinate of 1 atom
    HMM = sklearn.hmm.GaussianHMM(n_components=4)
    HMM.transmat_ = np.array([[0.9, 0.1, 0.0, 0.0],
                              [0.1, 0.7, 0.2, 0.0],
                              [0.0, 0.1, 0.8, 0.1],
                              [0.0, 0.1, 0.1, 0.8]])
    HMM.means_ = np.array([[-10, -10, -10],
                           [-5,  -5,   -5],
                           [5,    5,    5],
                           [10,  10,   10]])
    HMM.covars_ = np.array([[0.1, 0.1, 0.1],
                            [0.5, 0.5, 0.5],
                            [1, 1, 1],
                            [4, 4, 4]])
    HMM.startprob_ = np.array([1, 1, 1, 1]) / 4.0

    # get a 1 atom topology
    topology = md.load(get_mdtraj_fn('native.pdb')).restrict_atoms([1]).topology

    # generate the trajectories and save them to disk
    for i in range(10):
        d, s = HMM.sample(100)
        t = md.Trajectory(xyz=d.reshape(len(d), 1, 3), topology=topology)
        t.save(os.path.join(DATADIR, 'Trajectory%d.h5' % i))

    fetch_alanine_dipeptide()

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
    # Capture stdout
    split = shlex.split(str)
    print(str)
    with open(os.devnull, 'w') as noout:
        assert subprocess.call(split, stdout=noout) == 0

################################################################################
# Tests
################################################################################

def test_atomindices():
    fn = get_mdtraj_fn('2EQQ.pdb')
    t = md.load(fn)
    with tempdir():
        shell('msmb AtomIndices -o all.dat --all -a -p %s' % fn)
        shell('msmb AtomIndices -o all-pairs.dat --all -d -p %s' % fn)
        atoms = np.loadtxt('all.dat', int)
        pairs =  np.loadtxt('all-pairs.dat', int)
        eq(t.n_atoms, len(atoms))
        eq(int(t.n_atoms * (t.n_atoms-1) / 2), len(pairs))

    with tempdir():
        shell('msmb AtomIndices -o heavy.dat --heavy -a -p %s' % fn)
        shell('msmb AtomIndices -o heavy-pairs.dat --heavy -d -p %s' % fn)
        atoms = np.loadtxt('heavy.dat', int)
        pairs = np.loadtxt('heavy-pairs.dat', int)
        assert all(t.topology.atom(i).element.symbol != 'H' for i in atoms)
        assert sum(1 for a in t.topology.atoms if a.element.symbol != 'H') == len(atoms)
        eq(np.array(list(itertools.combinations(atoms, 2))), pairs)
    
    with tempdir():
        shell('msmb AtomIndices -o alpha.dat --alpha -a -p %s' % fn)
        shell('msmb AtomIndices -o alpha-pairs.dat --alpha -d -p %s' % fn)
        atoms = np.loadtxt('alpha.dat', int)
        pairs = np.loadtxt('alpha-pairs.dat', int)
        assert all(t.topology.atom(i).name == 'CA' for i in atoms)
        assert sum(1 for a in t.topology.atoms if a.name == 'CA') == len(atoms)
        eq(np.array(list(itertools.combinations(atoms, 2))), pairs)

    with tempdir():
        shell('msmb AtomIndices -o minimal.dat --minimal -a -p %s' % fn)
        shell('msmb AtomIndices -o minimal-pairs.dat --minimal -d -p %s' % fn)
        atoms = np.loadtxt('minimal.dat', int)
        pairs = np.loadtxt('minimal-pairs.dat', int)
        assert all(t.topology.atom(i).name in ['CA', 'CB', 'C', 'N' , 'O'] for i in atoms)
        eq(np.array(list(itertools.combinations(atoms, 2))), pairs)


def test_superpose_featurizer():
    with tempdir():
        shell('msmb AtomIndices -o all.dat --all -a -p %s/alanine_dipeptide/ala2.pdb' % get_data_home()),
        shell("msmb SuperposeFeaturizer --trjs '{data_home}/alanine_dipeptide/*.dcd'"
              " --out distances --atom_indices all.dat"
              " --reference_traj {data_home}/alanine_dipeptide/ala2.pdb"
              " --top {data_home}/alanine_dipeptide/ala2.pdb".format(
                  data_home=get_data_home()))
        ds = dataset('distances')
        assert len(ds) == 10
        assert ds[0].shape[1] == len(np.loadtxt('all.dat'))
        print(ds.provenance)


def test_atom_pairs_featurizer():
    with tempdir():
        shell('msmb AtomIndices -o all.dat --all -d -p %s/alanine_dipeptide/ala2.pdb' % get_data_home()),
        shell("msmb AtomPairsFeaturizer --trjs '{data_home}/alanine_dipeptide/*.dcd'"
              " --out pairs --pair_indices all.dat"
              " --top {data_home}/alanine_dipeptide/ala2.pdb".format(
                  data_home=get_data_home()))
        ds = dataset('pairs')
        assert len(ds) == 10
        assert ds[0].shape[1] == len(np.loadtxt('all.dat')**2)
        print(ds.provenance)


def test_help():
    shell('msmb -h')

"""
def test_fitghmm():
    with tempdir():
        RawPositionsFeaturizer().save('featurizer.pickl')
        shell('hmsm fit-ghmm --featurizer featurizer.pickl  --n-init 10  '
                  ' --n-states 4 --dir %s --ext h5 --top %s' % (
                      DATADIR, os.path.join(DATADIR, 'Trajectory0.h5')))
        shell('hmsm Inspect -i hmms.jsonlines --details')
        shell('hmsm sample-ghmm --no-match-vars -i hmms.jsonlines --lag-time 1 --n-state 4 '
              '--featurizer featurizer.pickl --dir %s --ext h5 --top %s' % (
                  DATADIR, os.path.join(DATADIR, 'Trajectory0.h5')))
        shell('hmsm means-ghmm -i hmms.jsonlines --lag-time 1 --n-state 4 '
              '--featurizer featurizer.pickl --dir %s --ext h5 --top %s' % (
                  DATADIR, os.path.join(DATADIR, 'Trajectory0.h5')))
        shell('hmsm structures means.csv --ext pdb --prefix means --top %s' %  os.path.join(DATADIR, 'Trajectory0.h5'))
        
        samples_csv = pd.read_csv('samples.csv', skiprows=1)
        means_csv = pd.read_csv('means.csv', skiprows=1)
        
        model = next(iterobjects('hmms.jsonlines'))
        means_pdb = md.load(glob('means-*.pdb'))

    means = np.array(sorted(model['means'], key=lambda e: e[0]))
    print('true\n', HMM.means_)
    print('learned\n', means)

    eq(HMM.means_, means, decimal=0)

    means_pdb_xyz = np.array(sorted(means_pdb.xyz.reshape(4, 3), key=lambda e: e[0]))
    eq(means_pdb_xyz, np.array(sorted(model['means'], key=lambda e:e[0])), decimal=0)
"""
