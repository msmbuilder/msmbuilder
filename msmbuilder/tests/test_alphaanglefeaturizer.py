import numpy as np
from mdtraj.testing import eq, raises
import msmbuilder.featurizer
from msmbuilder.featurizer import subset_featurizer
from msmbuilder.example_datasets import fetch_alanine_dipeptide

def test_AlanineDipeptide():
	# will produce 0 features because not enough peptides

	dataset = fetch_alanine_dipeptide()
	trajectories = dataset["trajectories"]
	trj0 = trajectories[0][0]
	featurizer = msmbuilder.featurizer.AlphaAngleFeaturizer()
	nothing = featurizer.transform(trajectories)

	assert(nothing[0].shape[1] == 0)

from msmbuilder.example_datasets import FsPeptide
fs_peptide = FsPeptide()
import tempfile
import os
os.chdir(tempfile.mkdtemp())
from msmbuilder.dataset import dataset

def test_FsPeptide():
	# will produce 36 features

	xyz = dataset(fs_peptide.data_dir + "/*.xtc", topology=fs_peptide.data_dir + '/fs_peptide.pdb')
	featurizer = msmbuilder.featurizer.AlphaAngleFeaturizer()
	alphas = xyz.create_derived('alphas_test/', fmt='dir-npy')
	for i in xyz.keys():
	    alphas[i] = featurizer.partial_transform(xyz[i])

	assert(alphas[0].shape[1] == 36)

def test_FsPeptide_nosincos():
	# will produce 18 features

	xyz = dataset(fs_peptide.data_dir + "/*.xtc", topology=fs_peptide.data_dir + '/fs_peptide.pdb')
	featurizer = msmbuilder.featurizer.AlphaAngleFeaturizer(sincos=False)
	alphas = xyz.create_derived('alphas_test_nosincos/', fmt='dir-npy')
	for i in xyz.keys():
	    alphas[i] = featurizer.partial_transform(xyz[i])

	assert(alphas[0].shape[1] == 18)






