# Author: Robert McGibbon, Kyle A. Beauchamp <rmcgibbo@gmail.com>
# Contributors:
# Copyright (c) 2014, Stanford University and the Authors
# All rights reserved.

# -----------------------------------------------------------------------------
# Imports
# -----------------------------------------------------------------------------
from __future__ import print_function, absolute_import, division

from glob import glob
from distutils.spawn import find_executable
import os
from os import makedirs, system
from os.path import exists, join, basename, expanduser
import tarfile
import tempfile
import itertools

import numpy as np
import mdtraj as md
from .base import Bunch, Dataset
from .base import get_data_home

def mae_to_pdb(in_filename, out_filename):
    """Use VMD to convert a meastro file into a PDB file."""
    
    tcl_text = """mol new %s
set sel [atomselect top all]
$sel writepdb %s
exit 0 
""" % (in_filename, out_filename)

    vmd_path = find_executable("vmd")

    if vmd_path is None:
        raise(RuntimeError("Cannot find vmd executable, which is required to parse DESRES datasets."))

    handle = tempfile.NamedTemporaryFile(suffix=".tcl")
    handle.write(tcl_text)
    handle.flush()

    script_name = handle.name
    cmd = "vmd -dispdev text -e %s" % script_name
    print(cmd)
    system(cmd)
    print("Done running vmd")
    handle.close()

class _DESRESDataset(Dataset):
    """Base class for DESRES tarball-based datasets

    Parameters
    ----------
    data_home : optional, default: None
        Specify another download and cache folder for the datasets. By default
        all MSMBuilder data is stored in '~/msmbuilder_data' subfolders.
    stride : int, optional, default=1
        How much to subsample the dataset.  

    """

    def __init__(self, data_home=None, stride=None):
        try:
            self._desres_tarball_path =  os.environ["DESRES_TARBALL_PATH"]
        except KeyError as e:
            raise(KeyError("Please set environment variable DESRES_TARBALL_PATH to point to path containing DESRES tarballs."))

        self.data_home = get_data_home(data_home)
        self.data_dir = join(self.data_home, self._target_directory)
        self.cached = False
        self._stride = stride  # How much to subsample trajectories, e.g. out_traj = traj[::stride]
        self.system_filename = join(self.data_dir, "system.pdb")
        self.protein_filename = join(self.data_dir, "protein.pdb")
        self._protein_indices = None  # Override this to manually specify which atoms to save.  If None, use mdtraj to select "protein"
        self._dcd_is_only_protein = True  # If False, dcds have protein + solvent
    
    def _get_dcd_filenames(self, archive):
        """Count the number of DCD files and return a list of their filenames."""
        filenames = []
        for i in itertools.count():
            filename = "%s/%s/%s-%.3d.dcd" % (self.top_dir, self.dcd_prefix, self.dcd_prefix, i)
            if filename in archive.getnames():
                filenames.append(filename)
            else:
                return filenames
    
    def cache(self):
        if not exists(self.data_home):
            makedirs(self.data_home)

        if not exists(self.data_dir):
            makedirs(self.data_dir)
            self._extract()
        self.cached = True

    def get(self):
        if not self.cached:
            self.cache()
        top = md.load(join(self.data_dir, self.protein_filename))
        trajectories = []
        for fn in sorted(glob(join(self.data_dir, 'trajectory*.xtc'))):
            print('loading %s...' % basename(fn))
            trajectories.append(md.load(fn, top=top))

        return Bunch(trajectories=trajectories, DESCR=self.description())

    def _extract(self):
        print("Extracting %s trajectory." % self.name)
        with md.utils.enter_temp_directory():
            print(os.getcwd())
            archive = tarfile.open(self._tarball_filename)  # , mode='r:gz')
            print("Extracting %s" % self._mae_filename)
            archive.extract(self._mae_filename)
            print("mae to pdb")
            mae_to_pdb(self._mae_filename, self.system_filename)
            print("md.load(%s)" % self.system_filename)
            traj0 = md.load(self.system_filename)
            
            if self._protein_indices is None:
                self._protein_indices = traj0.top.select("protein")
            
            traj0 = traj0.atom_slice(self._protein_indices)
            traj0.save(self.protein_filename)
            
            if not self._dcd_is_only_protein:
                traj0 = md.load(self.system_filename)
            
            filenames = self._get_dcd_filenames(archive)
            
            for filename in filenames:
                print(filename)
                archive.extract(filename)
            traj = md.load([filename for filename in filenames], top=traj0, stride=self._stride, atom_indices=self._protein_indices)
            out_filename = join(self.data_dir, "trajectory0.xtc")
            traj.save(out_filename)



class DESRES2F4K(_DESRESDataset):
    """HP35 125 us trajectory from DESRES 'How Fast Proteins Fold'.

    Parameters
    ----------
    data_home : optional, default: None
        Specify another download and cache folder for the datasets. By default
        all MSMBuilder data is stored in '~/msmbuilder_data' subfolders.
    stride : int, optional, default=5
        How much to subsample the dataset.  

    Notes
    -----
    You may be able to obtain this dataset by contacting DE Shaw research.
    Processing this dataset requires a working VMD installation, for
    converting the maestro file to PDB.
    """

    def __init__(self, data_home=None, stride=5, **kwargs):
        self._target_directory = "DESRES_2F4K_stride%d" % stride
        super(DESRES2F4K, self).__init__(data_home=data_home, stride=stride, **kwargs)
        self._tarball_filename = join(self._desres_tarball_path, "DESRES-Trajectory_2F4K-0-protein.tar.gz")
        self._mae_filename = "DESRES-Trajectory_2F4K-0-protein/system.mae"
        self.name = "DESRES 2F4K"
        self._protein_indices = np.arange(577)  # Have to manually enter because of some wacky atom names in MAE file
        self.top_dir = "DESRES-Trajectory_2F4K-0-protein"
        self.dcd_prefix = "2F4K-0-protein"


class DESRESBPTI(_DESRESDataset):
    """Millisecond BPTI trajectory from DESRES

    Parameters
    ----------
    data_home : optional, default: None
        Specify another download and cache folder for the datasets. By default
        all MSMBuilder data is stored in '~/msmbuilder_data' subfolders.
    stride : int, optional, default=1
        How much to subsample the dataset.  

    Notes
    -----
    You may be able to obtain this dataset by contacting DE Shaw research.
    Processing this dataset requires a working VMD installation, for
    converting the maestro file to PDB.
    """

    def __init__(self, data_home=None, stride=1, **kwargs):
        self._target_directory = "DESRES_BPTI_stride%d" % stride
        super(DESRESBPTI, self).__init__(data_home=data_home, stride=stride, **kwargs)
        self._tarball_filename = join(self._desres_tarball_path, "DESRES-Trajectory-bpti-100.tar.gz")
        self._mae_filename = "DESRES-Trajectory-bpti-100/bpti.mae"
        self.name = "DESRES BPTI"
        self.top_dir = "DESRES-Trajectory-bpti-100"
        self.dcd_prefix = "bpti-100" 
        self._dcd_is_only_protein = False


class DESRESFIP35_1(_DESRESDataset):
    """Trajectory 1 of DESRES FIP35 (Science 2010) dataset


    Parameters
    ----------
    data_home : optional, default: None
        Specify another download and cache folder for the datasets. By default
        all MSMBuilder data is stored in '~/msmbuilder_data' subfolders.
    stride : int, optional, default=5
        How much to subsample the dataset.  

    Notes
    -----
    You may be able to obtain this dataset by contacting DE Shaw research.
    Processing this dataset requires a working VMD installation, for
    converting the maestro file to PDB.
    """

    def __init__(self, data_home=None, stride=5, **kwargs):
        self.name = "DESRES_FIP35_1"
        self._target_directory = "%s_stride%d" % (self.name, stride)
        super(DESRESFIP35_1, self).__init__(data_home=data_home, stride=stride, **kwargs)
        self._tarball_filename = join(self._desres_tarball_path, "DESRES-Trajectory-ww_1-protein.tar")
        self._mae_filename = "DESRES-Trajectory-ww_1-protein/ww.mae"
        self._protein_indices = np.arange(528)  # Have to manually enter because of some wacky atom names / numbering in MAE file
        self.top_dir = "DESRES-Trajectory-ww_1-protein"
        self.dcd_prefix = "ww_1-protein" 
        self._dcd_is_only_protein = True

class DESRESFIP35_2(_DESRESDataset):
    """Trajectory 1 of DESRES FIP35 (Science 2010) dataset

    Parameters
    ----------
    data_home : optional, default: None
        Specify another download and cache folder for the datasets. By default
        all MSMBuilder data is stored in '~/msmbuilder_data' subfolders.
    stride : int, optional, default=5
        How much to subsample the dataset.  

    Notes
    -----
    You may be able to obtain this dataset by contacting DE Shaw research.
    Processing this dataset requires a working VMD installation, for
    converting the maestro file to PDB.
    """

    def __init__(self, data_home=None, stride=5, **kwargs):
        self.name = "DESRES_FIP35_2"
        self._target_directory = "%s_stride%d" % (self.name, stride)
        super(DESRESFIP35_2, self).__init__(data_home=data_home, stride=stride, **kwargs)
        self._tarball_filename = join(self._desres_tarball_path, "DESRES-Trajectory-ww_2-protein.tar")
        self._mae_filename = "DESRES-Trajectory-ww_2-protein/ww.mae"
        self._protein_indices = np.arange(528)  # Have to manually enter because of some wacky atom names / numbering in MAE file
        self.top_dir = "DESRES-Trajectory-ww_2-protein"
        self.dcd_prefix = "ww_2-protein" 
        self._dcd_is_only_protein = True


def fetch_2f4k(data_home=None):
    return DESRES2F4K(data_home).get()

fetch_2f4k.__doc__ = DESRES2F4K.__doc__

def fetch_bpti(data_home=None):
    return DESRESBPTI(data_home).get()

fetch_bpti.__doc__ = DESRESBPTI.__doc__

def fetch_fip35_1(data_home=None):
    return DESRESFIP35_1(data_home).get()

fetch_fip35_1.__doc__ = DESRESFIP35_1.__doc__

def fetch_fip35_2(data_home=None):
    return DESRESFIP35_2(data_home).get()

fetch_fip35_2.__doc__ = DESRESFIP35_2.__doc__
