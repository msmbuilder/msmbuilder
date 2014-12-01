# Author: Robert McGibbon <rmcgibbo@gmail.com>
# Contributors:
# Copyright (c) 2014, Stanford University and the Authors
# All rights reserved.

# -----------------------------------------------------------------------------
# Imports
# -----------------------------------------------------------------------------
from __future__ import print_function, absolute_import, division

from glob import glob
from io import BytesIO
from os import makedirs
from os.path import exists
from os.path import join
from os.path import basename
from zipfile import ZipFile

try:
    # Python 2
    from urllib2 import urlopen
except ImportError:
    # Python 3+
    from urllib.request import urlopen

import mdtraj as md
from .base import Bunch, Dataset
from .base import get_data_home

DATA_URL = "http://downloads.figshare.com/article/public/1030363"
TARGET_DIRECTORY = "fs_peptide"


class FsPeptide(Dataset):
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

    def __init__(self, data_home=None):
        self.data_home = get_data_home(data_home)
        self.data_dir = join(self.data_home, TARGET_DIRECTORY)
        self.cached = False

    def cache(self):
        if not exists(self.data_home):
            makedirs(self.data_home)

        if not exists(self.data_dir):
            print('downloading fs peptide from %s to %s' %
                  (DATA_URL, self.data_home))
            fhandle = urlopen(DATA_URL)
            buf = BytesIO(fhandle.read())
            zip_file = ZipFile(buf)
            makedirs(self.data_dir)
            for name in zip_file.namelist():
                zip_file.extract(name, path=self.data_dir)

        self.cached = True

    def get(self):
        if not self.cached:
            self.cache()
        top = md.load(join(self.data_dir, 'fs_peptide.pdb'))
        trajectories = []
        for fn in sorted(glob(join(self.data_dir, 'trajectory*.xtc'))):
            print('loading %s...' % basename(fn))
            trajectories.append(md.load(fn, top=top))

        return Bunch(trajectories=trajectories, DESCR=self.description())


def fetch_fs_peptide(data_home=None):
    return FsPeptide(data_home).get()


fetch_fs_peptide.__doc__ = FsPeptide.__doc__
