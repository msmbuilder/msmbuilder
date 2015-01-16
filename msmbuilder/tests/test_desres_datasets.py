from __future__ import print_function, absolute_import, division
import os
import shutil
import tempfile
from six.moves import cPickle

import numpy as np
from nose.tools import assert_raises
from unittest import skipIf
from msmbuilder.example_datasets import fetch_fip35_1, fetch_fip35_2, fetch_2f4k, fetch_bpti
from mdtraj.testing import get_fn
from sklearn.externals.joblib import Parallel, delayed

from .test_commands import tempdir

try:
    DESRES_TARBALL_PATH =  os.environ["DESRES_TARBALL_PATH"]
except KeyError:
    DESRES_TARBALL_PATH = None

@skipIf(DESRES_TARBALL_PATH is None, "Skipping DESRES dataset tests.")
def test_fip35_1():
    path = tempfile.mkdtemp()
    try:
        ds = fetch_fip35_1(path)
        trajectories = ds["trajectories"]
        traj = trajectories[0]
        assert(len(trajectories) == 1)
        assert(traj.n_atoms == 528)
        assert(traj.n_frames == 100000)

    except:
        raise
    finally:
        shutil.rmtree(path)


@skipIf(DESRES_TARBALL_PATH is None, "Skipping DESRES dataset tests.")
def test_fip35_2():
    path = tempfile.mkdtemp()
    try:
        ds = fetch_fip35_2(path)
        trajectories = ds["trajectories"]
        traj = trajectories[0]
        assert(len(trajectories) == 1)
        assert(traj.n_atoms == 528)
        assert(traj.n_frames == 100000)

    except:
        raise
    finally:
        shutil.rmtree(path)


@skipIf(DESRES_TARBALL_PATH is None, "Skipping DESRES dataset tests.")
def test_2f4k():
    path = tempfile.mkdtemp()
    try:
        ds = fetch_2f4k(path)
        trajectories = ds["trajectories"]
        traj = trajectories[0]
        assert(len(trajectories) == 1)
        assert(traj.n_atoms == 577)
        assert(traj.n_frames == 125582)

    except:
        raise
    finally:
        shutil.rmtree(path)


@skipIf(DESRES_TARBALL_PATH is None, "Skipping DESRES dataset tests.")
def test_bpti():
    path = tempfile.mkdtemp()
    try:
        ds = fetch_bpti(path)
        trajectories = ds["trajectories"]
        traj = trajectories[0]
        assert(len(trajectories) == 1)
        assert(traj.n_atoms == 892)
        assert(traj.n_frames == 41250)

    except:
        raise
    finally:
        shutil.rmtree(path)
