# Author: Robert McGibbon <rmcgibbo@gmail.com>
# Contributors: Matthew Harrigan <matthew.harrigan@outlook.com>
# Copyright (c) 2016, Stanford University and the Authors
# All rights reserved.

# -----------------------------------------------------------------------------
# Imports
# -----------------------------------------------------------------------------
from __future__ import print_function, absolute_import, division

from glob import glob
from os.path import basename
from os.path import join

import mdtraj as md

from .base import Bunch, _MDDataset

DATA_URL = "https://ndownloader.figshare.com/articles/1030363/versions/1"
TARGET_DIRECTORY = "fs_peptide"


class FsPeptide(_MDDataset):
    """Fs peptide (implicit solvent) dataset

    Parameters
    ----------
    data_home : optional, default: None
        Specify another download and cache folder for the datasets. By default
        all MSMBuilder data is stored in '~/msmbuilder_data' subfolders.

    Notes
    -----
    This dataset consists of 28 molecular dynamics trajectories of Fs peptide
    (Ace-A_5(AAARA)_3A-NME), a widely studied model system for protein folding.
    Each trajectory is 500 ns in length, and saved at a 50 ps time interval (14
    us aggegrate sampling). The simulations were performed using the AMBER99SB-ILDN
    force field with GBSA-OBC implicit solvent at 300K, starting from randomly
    sampled conformations from an initial 400K unfolding simulation. The
    simulations were performed with OpenMM 6.0.1.

    The dataset, including the script used to generate the dataset
    is available on figshare at

        http://dx.doi.org/10.6084/m9.figshare.1030363
    """

    target_directory = TARGET_DIRECTORY
    data_url = DATA_URL

    def get_cached(self):
        try:
            top = md.load(join(self.data_dir, 'fs-peptide.pdb'))
        except IOError:
            top = md.load(join(self.data_dir, 'fs_peptide.pdb'))
        trajectories = []
        for fn in sorted(glob(join(self.data_dir, 'trajectory*.xtc'))):
            if self.verbose:
                print('loading %s...' % basename(fn))
            trajectories.append(md.load(fn, top=top))

        return Bunch(trajectories=trajectories, DESCR=self.description())


class MinimalFsPeptide(_MDDataset):
    """Minimal Fs peptide (implicit solvent) dataset

    Notes
    -----
    This dataset consists of 28 molecular dynamics trajectories of Fs peptide
    (Ace-A_5(AAARA)_3A-NME), a widely studied model system for protein folding.
    Each trajectory is 500 ns in length, and saved at a 10 ns time interval (14
    us aggegrate sampling). The simulations were performed using the AMBER99SB-ILDN
    force field with GBSA-OBC implicit solvent at 300K, starting from randomly
    sampled conformations from an initial 400K unfolding simulation. The
    simulations were performed with OpenMM 6.0.1.

    The dataset is a subsampling of the FsPeptide dataset described on
    figshare at

        http://dx.doi.org/10.6084/m9.figshare.1030363
    """

    target_directory = TARGET_DIRECTORY
    data_url = DATA_URL

    def get_cached(self):
        try:
            top = md.load(join(self.data_dir, 'fs-peptide.pdb'))
        except IOError:
            top = md.load(join(self.data_dir, 'fs_peptide.pdb'))
        trajectories = [
            md.load(fn, top=top, stride=200)
            for fn in sorted(glob("{}/trajectory*.xtc".format(self.data_dir)))
            ]
        return Bunch(trajectories=trajectories, DESCR=self.description())


def fetch_fs_peptide(data_home=None):
    return FsPeptide(data_home).get()


fetch_fs_peptide.__doc__ = FsPeptide.__doc__
