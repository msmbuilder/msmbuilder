import numpy as np
import mdtraj as md
import pandas as pd
import glob
import os
from scipy.stats import vonmises as vm

from msmbuilder.example_datasets import fetch_fs_peptide,FsPeptide
from msmbuilder.featurizer import DihedralFeaturizer, AlphaAngleFeaturizer,\
    KappaAngleFeaturizer,ContactFeaturizer,VonMisesFeaturizer


"""
Series of tests to make sure all the describe features are putting the right features in the
right place
"""

fs = FsPeptide()
fs.cache()
dirname = fs.data_dir
top = md.load(dirname+"/fs-peptide.pdb")

if np.random.choice([True, False]):
    atom_ind=[i.index for i in top.top.atoms if i.residue.is_protein
          and (i.residue.index in range(15) or i.residue.index in range(20,23))]
else:
    atom_ind =[i.index for i in top.top.atoms]

trajectories = [md.load(fn, stride=100, top=top,atom_indices=atom_ind) for fn in
                glob.glob(os.path.join(dirname,"trajectory*.xtc"))]

def test_DihedralFeaturizer_describe_features():

    feat = DihedralFeaturizer()

    rnd_traj = np.random.randint(len(trajectories))

    features = feat.transform([trajectories[rnd_traj]])

    df = pd.DataFrame(feat.describe_features(trajectories[rnd_traj]))

    for f in range(25):
        f_index = np.random.choice(len(df))

        atom_inds = df.iloc[f_index].atominds
        feature_value = md.compute_dihedrals(trajectories[rnd_traj],[atom_inds])
        if feat.sincos:
            func = getattr(np, '%s' %df.iloc[f_index].otherinfo)
            feature_value = func(feature_value)

        assert (features[0][:,f_index]==feature_value.flatten()).all()

def test_DihedralFeaturizer_describe_features_nosincos():

    feat = DihedralFeaturizer(sincos=False)

    rnd_traj = np.random.randint(len(trajectories))

    features = feat.transform([trajectories[rnd_traj]])

    df = pd.DataFrame(feat.describe_features(trajectories[rnd_traj]))

    for f in range(25):
        f_index = np.random.choice(len(df))

        atom_inds = df.iloc[f_index].atominds
        feature_value = md.compute_dihedrals(trajectories[rnd_traj],[atom_inds])
        if feat.sincos:
            func = getattr(np, '%s' %df.iloc[f_index].otherinfo)
            feature_value = func(feature_value)

        assert (features[0][:,f_index]==feature_value.flatten()).all()

def test_AlphaFeaturizer_describe_features():

    feat = AlphaAngleFeaturizer()

    rnd_traj = np.random.randint(len(trajectories))

    features = feat.transform([trajectories[rnd_traj]])

    df = pd.DataFrame(feat.describe_features(trajectories[rnd_traj]))

    for f in range(25):
        f_index = np.random.choice(len(df))

        atom_inds = df.iloc[f_index].atominds

        feature_value = md.compute_dihedrals(trajectories[rnd_traj],[atom_inds])
        if feat.sincos:
            func = getattr(np, '%s' %df.iloc[f_index].otherinfo)
            feature_value = func(feature_value)

        assert (features[0][:,f_index] == feature_value.flatten()).all()

def test_AlphaFeaturizer_describe_features_nosincos():

    feat = AlphaAngleFeaturizer(sincos=False)

    rnd_traj = np.random.randint(len(trajectories))

    features = feat.transform([trajectories[rnd_traj]])

    df = pd.DataFrame(feat.describe_features(trajectories[rnd_traj]))

    for f in range(25):
        f_index = np.random.choice(len(df))

        atom_inds = df.iloc[f_index].atominds

        feature_value = md.compute_dihedrals(trajectories[rnd_traj],[atom_inds])
        if feat.sincos:
            func = getattr(np, '%s' %df.iloc[f_index].otherinfo)
            feature_value = func(feature_value)

        assert (features[0][:,f_index] == feature_value.flatten()).all()

def test_KappaFeaturizer_describe_features():

    feat = KappaAngleFeaturizer()

    rnd_traj = np.random.randint(len(trajectories))

    features = feat.transform([trajectories[rnd_traj]])

    df = pd.DataFrame(feat.describe_features(trajectories[rnd_traj]))

    for f in range(25):
        f_index = np.random.choice(len(df))

        atom_inds = df.iloc[f_index].atominds

        feature_value = md.compute_angles(trajectories[rnd_traj],[atom_inds])
        if feat.cos:
            func = getattr(np, '%s' %df.iloc[f_index].otherinfo)
            feature_value = func(feature_value)

        assert (features[0][:,f_index] == feature_value.flatten()).all()

def test_VonMisesFeaturizer_describe_features():

    feat = VonMisesFeaturizer()

    rnd_traj = np.random.randint(len(trajectories))

    features = feat.transform([trajectories[rnd_traj]])

    df = pd.DataFrame(feat.describe_features(trajectories[rnd_traj]))

    for f in range(25):
        f_index = np.random.choice(len(df))

        atom_inds = df.iloc[f_index].atominds
        bin_index = int(df.iloc[f_index].otherinfo.strip('bin-'))

        dihedral_value = md.compute_dihedrals(trajectories[rnd_traj],[atom_inds])

        feature_value = [vm.pdf(i, loc=feat.loc, kappa=feat.kappa)[bin_index]
                         for i in dihedral_value]

        assert (features[0][:,f_index] == feature_value).all()

def test_ContactFeaturizer_describe_features():

    feat = ContactFeaturizer(scheme='CA',ignore_nonprotein=True)

    rnd_traj = np.random.randint(len(trajectories))

    features = feat.transform([trajectories[rnd_traj]])

    df = pd.DataFrame(feat.describe_features(trajectories[rnd_traj]))

    for f in range(25):
        f_index = np.random.choice(len(df))

        residue_ind = df.iloc[f_index].resids

        feature_value,_ = md.compute_contacts(trajectories[rnd_traj],contacts=[residue_ind],
                                            scheme='ca',ignore_nonprotein=True,)

        assert (features[0][:,f_index] == feature_value.flatten()).all()