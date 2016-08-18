"""Make sure all the describe features are putting the right features in the
right place
"""
import mdtraj as md
import numpy as np
import pandas as pd
from mdtraj.testing import eq
from scipy.stats import vonmises as vm

from msmbuilder.example_datasets import MinimalFsPeptide
from msmbuilder.feature_selection import FeatureSelector
from msmbuilder.featurizer import DihedralFeaturizer, AlphaAngleFeaturizer, \
    KappaAngleFeaturizer, ContactFeaturizer, VonMisesFeaturizer

trajectories = MinimalFsPeptide().get_cached().trajectories
top = trajectories[0].topology

if np.random.choice([True, False]):
    atom_ind = [i.index for i in top.atoms
                if i.residue.is_protein
                and (i.residue.index in range(15)
                     or i.residue.index in range(20, 23))]
else:
    atom_ind = [i.index for i in top.atoms]


def test_DihedralFeaturizer_describe_features():
    feat = DihedralFeaturizer()
    rnd_traj = np.random.randint(len(trajectories))
    features = feat.transform([trajectories[rnd_traj]])
    df = pd.DataFrame(feat.describe_features(trajectories[rnd_traj]))

    for f in range(25):
        f_index = np.random.choice(len(df))

        atom_inds = df.iloc[f_index].atominds
        feature_value = md.compute_dihedrals(trajectories[rnd_traj],
                                             [atom_inds])
        if feat.sincos:
            func = getattr(np, '%s' % df.iloc[f_index].otherinfo)
            feature_value = func(feature_value)

        assert (features[0][:, f_index] == feature_value.flatten()).all()


def test_DihedralFeaturizer_describe_features_nosincos():
    feat = DihedralFeaturizer(sincos=False)
    rnd_traj = np.random.randint(len(trajectories))
    features = feat.transform([trajectories[rnd_traj]])
    df = pd.DataFrame(feat.describe_features(trajectories[rnd_traj]))

    for f in range(25):
        f_index = np.random.choice(len(df))

        atom_inds = df.iloc[f_index].atominds
        feature_value = md.compute_dihedrals(trajectories[rnd_traj],
                                             [atom_inds])
        if feat.sincos:
            func = getattr(np, '%s' % df.iloc[f_index].otherinfo)
            feature_value = func(feature_value)

        assert (features[0][:, f_index] == feature_value.flatten()).all()


def test_AlphaFeaturizer_describe_features():
    feat = AlphaAngleFeaturizer()
    rnd_traj = np.random.randint(len(trajectories))
    features = feat.transform([trajectories[rnd_traj]])
    df = pd.DataFrame(feat.describe_features(trajectories[rnd_traj]))

    for f in range(25):
        f_index = np.random.choice(len(df))

        atom_inds = df.iloc[f_index].atominds

        feature_value = md.compute_dihedrals(trajectories[rnd_traj],
                                             [atom_inds])
        if feat.sincos:
            func = getattr(np, '%s' % df.iloc[f_index].otherinfo)
            feature_value = func(feature_value)

        assert (features[0][:, f_index] == feature_value.flatten()).all()


def test_AlphaFeaturizer_describe_features_nosincos():
    feat = AlphaAngleFeaturizer(sincos=False)
    rnd_traj = np.random.randint(len(trajectories))
    features = feat.transform([trajectories[rnd_traj]])
    df = pd.DataFrame(feat.describe_features(trajectories[rnd_traj]))

    for f in range(25):
        f_index = np.random.choice(len(df))

        atom_inds = df.iloc[f_index].atominds

        feature_value = md.compute_dihedrals(trajectories[rnd_traj],
                                             [atom_inds])
        if feat.sincos:
            func = getattr(np, '%s' % df.iloc[f_index].otherinfo)
            feature_value = func(feature_value)

        assert (features[0][:, f_index] == feature_value.flatten()).all()


def test_KappaFeaturizer_describe_features():
    feat = KappaAngleFeaturizer()
    rnd_traj = np.random.randint(len(trajectories))
    features = feat.transform([trajectories[rnd_traj]])
    df = pd.DataFrame(feat.describe_features(trajectories[rnd_traj]))

    for f in range(25):
        f_index = np.random.choice(len(df))

        atom_inds = df.iloc[f_index].atominds

        feature_value = md.compute_angles(trajectories[rnd_traj], [atom_inds])
        if feat.cos:
            func = getattr(np, '%s' % df.iloc[f_index].otherinfo)
            feature_value = func(feature_value)

        assert (features[0][:, f_index] == feature_value.flatten()).all()


def test_VonMisesFeaturizer_describe_features():
    feat = VonMisesFeaturizer()
    rnd_traj = np.random.randint(len(trajectories))
    features = feat.transform([trajectories[rnd_traj]])
    df = pd.DataFrame(feat.describe_features(trajectories[rnd_traj]))

    for f in range(25):
        f_index = np.random.choice(len(df))

        atom_inds = df.iloc[f_index].atominds
        bin_index = int(df.iloc[f_index].otherinfo.strip('bin-'))

        dihedral_value = md.compute_dihedrals(trajectories[rnd_traj],
                                              [atom_inds])

        feature_value = [vm.pdf(i, loc=feat.loc, kappa=feat.kappa)[bin_index]
                         for i in dihedral_value]

        assert (features[0][:, f_index] == feature_value).all()


def test_ContactFeaturizer_describe_features():
    feat = ContactFeaturizer(scheme='CA', ignore_nonprotein=True)
    rnd_traj = np.random.randint(len(trajectories))
    features = feat.transform([trajectories[rnd_traj]])
    df = pd.DataFrame(feat.describe_features(trajectories[rnd_traj]))

    for f in range(25):
        f_index = np.random.choice(len(df))

        residue_ind = df.iloc[f_index].resids

        feature_value, _ = md.compute_contacts(trajectories[rnd_traj],
                                               contacts=[residue_ind],
                                               scheme='ca',
                                               ignore_nonprotein=True, )

        assert (features[0][:, f_index] == feature_value.flatten()).all()


def test_FeatureSelector_describe_features():
    rnd_traj = np.random.randint(len(trajectories))
    f_ca = ContactFeaturizer(scheme='CA', ignore_nonprotein=True)
    f1 = f_ca.transform([trajectories[rnd_traj]])
    df1 = pd.DataFrame(f_ca.describe_features(trajectories[rnd_traj]))

    f_dih = DihedralFeaturizer()
    f2 = f_dih.transform([trajectories[rnd_traj]])
    df2 = pd.DataFrame(f_dih.describe_features(trajectories[rnd_traj]))

    df_dict = {}
    df_dict["ca"] = df1
    df_dict["dih"] = df2

    f_comb = FeatureSelector([('ca', f_ca), ('dih', f_dih)])
    f3 = f_comb.transform([trajectories[rnd_traj]])
    df3 = pd.DataFrame(f_comb.describe_features(trajectories[rnd_traj]))
    assert len(df3) == len(df1) + len(df2)
    df4 = pd.concat([df_dict[i] for i in f_comb.feat_list])
    # lets randomly compare 40 features
    for i in np.random.choice(range(len(df3)), 40):
        for j in df3.columns:
            assert eq(df3.iloc[i][j], df4.iloc[i][j])
