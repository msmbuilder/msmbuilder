# Author: Robert McGibbon <rmcgibbo@gmail.com>
# Contributors:
# Copyright (c) 2014, Stanford University and the Authors
# All rights reserved.
#
# Mixtape is free software: you can redistribute it and/or modify
# it under the terms of the GNU Lesser General Public License as
# published by the Free Software Foundation, either version 2.1
# of the License, or (at your option) any later version.
#
# This library is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Lesser General Public License for more details.
#
# You should have received a copy of the GNU Lesser General Public
# License along with Mixtape. If not, see <http://www.gnu.org/licenses/>.

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

try:
    # Python 2
    from urllib2 import urlopen
except ImportError:
    # Python 3+
    from urllib.request import urlopen

import mdtraj as md
from .base import Bunch, Dataset
from .base import get_data_home

DATA_URL = "http://downloads.figshare.com/article/public/1026131"
TARGET_DIRECTORY = "alanine_dipeptide"


class AlanineDipeptide(Dataset):
    """Alanine dipeptide dataset

    Parameters
    ----------
    data_home : optional, default: None
        Specify another download and cache folder for the datasets. By default
        all mixtape data is stored in '~/mixtape_data' subfolders.


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

    def get(self):
        if not self.cached:
            self.cache()

        top = md.load(join(self.data_dir, 'ala2.pdb'))
        trajectories = []
        for fn in glob(join(self.data_dir, 'trajectory*.dcd')):
            trajectories.append(md.load(fn, top=top))

        return Bunch(trajectories=trajectories, DESCR=self.description())


def fetch_alanine_dipeptide(data_home=None):
    return AlanineDipeptide(data_home).get()


fetch_alanine_dipeptide.__doc__ = AlanineDipeptide.__doc__

