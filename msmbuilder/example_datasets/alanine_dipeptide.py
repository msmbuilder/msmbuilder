# Author: Robert McGibbon <rmcgibbo@gmail.com>
# Contributors: Matthew Harrigan <matthew.harrigan@outlook.com>
# Copyright (c) 2016, Stanford University and the Authors
# All rights reserved.

# -----------------------------------------------------------------------------
# Imports
# -----------------------------------------------------------------------------
from __future__ import print_function, absolute_import, division

from glob import glob
from os.path import join

import mdtraj as md

from .base import Bunch, _MDDataset

DATA_URL = "https://ndownloader.figshare.com/articles/1026131/versions/8"
TARGET_DIRECTORY = "alanine_dipeptide"


class AlanineDipeptide(_MDDataset):
    """Alanine dipeptide dataset

    Parameters
    ----------
    data_home : optional, default: None
        Specify another download and cache folder for the datasets. By default
        all MSMBuilder data is stored in '~/msmbuilder_data' subfolders.


    Notes
    -----
    The dataset consists of ten 10ns trajectories of of alanine dipeptide,
    simulated using OpenMM 6.0.1 (CUDA platform, NVIDIA GTX660) with the
    AMBER99SB-ILDN force field at 300K (langevin dynamics, friction coefficient
    of 91/ps, timestep of 2fs) with GBSA implicit solvent. The coordinates are
    saved every 1ps. Each trajectory contains 9,999 snapshots.

    The dataset, including the script used to generate the dataset
    is available on figshare at

        http://dx.doi.org/10.6084/m9.figshare.1026131
    """
    target_directory = TARGET_DIRECTORY
    data_url = DATA_URL

    def get_cached(self):
        top = md.load(join(self.data_dir, 'ala2.pdb'))
        trajectories = []
        for fn in glob(join(self.data_dir, 'trajectory*.dcd')):
            trajectories.append(md.load(fn, top=top))

        return Bunch(trajectories=trajectories, DESCR=self.description())


def fetch_alanine_dipeptide(data_home=None):
    return AlanineDipeptide(data_home).get()


fetch_alanine_dipeptide.__doc__ = AlanineDipeptide.__doc__
