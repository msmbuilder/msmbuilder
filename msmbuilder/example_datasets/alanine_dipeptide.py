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
from zipfile import ZipFile
from six.moves.urllib.request import urlopen

import mdtraj as md
from .base import Bunch, Dataset
from .base import get_data_home, retry

DATA_URL = "https://ndownloader.figshare.com/articles/1026131/versions/8"
TARGET_DIRECTORY = "alanine_dipeptide"


class AlanineDipeptide(Dataset):
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

    def __init__(self, data_home=None):
        self.data_home = get_data_home(data_home)
        self.data_dir = join(self.data_home, TARGET_DIRECTORY)
        self.cached = False

    @retry(3)
    def cache(self):
        if not exists(self.data_home):
            makedirs(self.data_home)

        if not exists(self.data_dir):
            print('downloading alanine dipeptide from %s to %s' %
                  (DATA_URL, self.data_home))
            fhandle = urlopen(DATA_URL)
            buf = BytesIO(fhandle.read())
            zip_file = ZipFile(buf)
            makedirs(self.data_dir)
            for name in zip_file.namelist():
                zip_file.extract(name, path=self.data_dir)

        self.cached = True

    def get_cached(self):
        top = md.load(join(self.data_dir, 'ala2.pdb'))
        trajectories = []
        for fn in glob(join(self.data_dir, 'trajectory*.dcd')):
            trajectories.append(md.load(fn, top=top))

        return Bunch(trajectories=trajectories, DESCR=self.description())

    def get(self):
        if not self.cached:
            self.cache()
        return self.get_cached()


def fetch_alanine_dipeptide(data_home=None):
    return AlanineDipeptide(data_home).get()


fetch_alanine_dipeptide.__doc__ = AlanineDipeptide.__doc__
