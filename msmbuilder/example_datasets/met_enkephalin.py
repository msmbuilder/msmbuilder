# Author: Robert McGibbon <rmcgibbo@gmail.com>
# Contributors:
# Copyright (c) 2014, Stanford University and the Authors
# All rights reserved.

# -----------------------------------------------------------------------------
# Imports
# -----------------------------------------------------------------------------
from __future__ import print_function, absolute_import, division

from glob import glob
from os.path import join

import mdtraj as md

from .base import Bunch, _MDDataset

DATA_URL = "https://ndownloader.figshare.com/articles/1026324/versions/1"
TARGET_DIRECTORY = "met_enkephalin"


class MetEnkephalin(_MDDataset):
    """Loader for the met-enkephalin dataset

    Parameters
    ----------
    data_home : optional, default: None
        Specify another download and cache folder for the datasets. By default
        all MSMBuilder data is stored in '~/msmbuilder_data' subfolders.

    download_if_missing: optional, True by default
        If False, raise a IOError if the data is not locally available
        instead of trying to download the data from the source site.

    Notes
    -----
    The dataset consists of ten ~50 ns molecular dynamics (MD) simulation
    trajectories of the 5 residue Met-enkaphalin peptide. The aggregate
    sampling is 499.58 ns. Simulations were performed starting from the 1st
    model in the 1PLX PDB file, solvated with 832 TIP3P water molecules using
    OpenMM 6.0. The coordinates (protein only -- the water was stripped)
    are saved every 5 picoseconds. Each of the ten trajectories is roughly
    50 ns long and contains about 10,000 snapshots.

    Forcefield: amber99sb-ildn; water: tip3p; nonbonded method: PME; cutoffs:
    1nm; bonds to hydrogen were constrained; integrator: langevin dynamics;
    temperature: 300K; friction coefficient: 1.0/ps; pressure control: Monte
    Carlo barostat (interval of 25 steps); timestep 2 fs.

    The dataset is available on figshare at

    http://dx.doi.org/10.6084/m9.figshare.1026324
    """

    data_url = DATA_URL
    target_directory = TARGET_DIRECTORY

    def get_cached(self):
        top = md.load(join(self.data_dir, '1plx.pdb'))
        trajectories = []
        for fn in glob(join(self.data_dir, 'trajectory*.dcd')):
            trajectories.append(md.load(fn, top=top))

        return Bunch(trajectories=trajectories, DESCR=self.description())


def fetch_met_enkephalin(data_home=None):
    return MetEnkephalin(data_home).get()


fetch_met_enkephalin.__doc__ = MetEnkephalin.__doc__
