import numpy as np
import mdtraj as md
from mdtraj import compute_dihedrals, compute_phi
from mdtraj.testing import eq
import pandas as pd
from scipy.stats import vonmises as vm

from msmbuilder.example_datasets import fetch_fs_peptide
from msmbuilder.featurizer import DihedralFeaturizer, AlphaAngleFeaturizer,\
    KappaAngleFeaturizer,ContactFeaturizer,VonMisesFeaturizer


"""
Series of tests to make sure all the describe features are putting the right features in the
right place
"""
dataset = fetch_fs_peptide()
trajectories = dataset["trajectories"]

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