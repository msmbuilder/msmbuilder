# Author: Matthew Harrigan <matthew.harrigan@outlook.com>
# Contributors:
# Copyright (c) 2016, Stanford University
# All rights reserved.

import os
import pickle
import shutil
import warnings

import mdtraj as md
import numpy as np
import pandas as pd


def backup(fn):
    if not os.path.exists(fn):
        return

    backnum = 1
    backfmt = "{fn}.bak.{backnum}"
    trial_fn = backfmt.format(fn=fn, backnum=backnum)
    while os.path.exists(trial_fn):
        backnum += 1
        trial_fn = backfmt.format(fn=fn, backnum=backnum)

    warnings.warn("{fn} exists. Moving it to {newfn}"
                  .format(fn=fn, newfn=trial_fn))
    shutil.move(fn, trial_fn)


def get_fn(base_fn, key):
    dfmt = "{}"
    ffmt = "{}.npy"
    if isinstance(key, tuple):  # TODO: check multiindex
        paths = [dfmt.format(k) for k in key[:-1]]
        paths += [ffmt.format(key[-1])]
        return os.path.join(base_fn, *paths)
    return os.path.join(base_fn, ffmt.format(key))


def preload_tops(meta):
    """Load all topology files into memory.

    This might save some performance compared to re-parsing the topology
    file for each trajectory you try to load in. Typically, you have far
    fewer (possibly 1) topologies than trajectories

    Parameters
    ----------
    meta : pd.DataFrame
        The DataFrame of metadata with a column named 'top_fn'

    Returns
    -------
    tops : dict
        Dictionary of ``md.Topology`` objects, keyed by "top_fn"
        values.
    """
    top_fns = set(meta['top_fn'])
    tops = {}
    for tfn in top_fns:
        tops[tfn] = md.load(tfn)
    return tops


def preload_top(meta):
    """Load one topology file into memory.

    This function checks to make sure there's only one topology file
    in play. When sampling frames, you have to have all the same
    topology to concatenate.

    Parameters
    ----------
    meta : pd.DataFrame
        The DataFrame of metadata with a column named 'top_fn'

    Returns
    -------
    top : md.Topology
        The one topology file that can be used for all trajectories.
    """
    top_fns = set(meta['top_fn'])
    if len(top_fns) != 1:
        raise ValueError("More than one topology is used in this project!")
    return md.load(top_fns.pop())


def load_meta(meta_fn='meta.pandas.pickl'):
    """Load metadata associated with a project.

    Parameters
    ----------
    meta_fn : str
        The filename

    Returns
    -------
    meta : pd.DataFrame
        Pandas DataFrame where each row contains metadata for a
        trajectory.
    """
    return pd.read_pickle(meta_fn)


def save_meta(meta, meta_fn='meta.pandas.pickl'):
    """Save metadata associated with a project.

    Parameters
    ----------
    meta : pd.DataFrame
        The DataFrame of metadata
    meta_fn : str
        The filename
    """
    pd.to_pickle(meta, meta_fn)


def save_generic(obj, fn):
    """Save Python objects, including msmbuilder Estimators.

    This is a convenience wrapper around Python's ``pickle``
    serialization scheme. This protocol is backwards-compatible
    among Python versions, but may not be "forwards-compatible".
    A file saved with Python 3 won't be able to be opened under Python 2.
    Please read the pickle docs (specifically related to the ``protocol``
    parameter) to specify broader compatibility.

    If a file already exists at the given filename, it will be backed
    up.

    Parameters
    ----------
    obj : object
        A Python object to serialize (save to disk)
    fn : str
        Filename to save the object. We recommend using the '.pickl'
        extension, but don't do anything to enforce that convention.
    """
    backup(fn)
    with open(fn, 'wb') as f:
        pickle.dump(obj, f)


def load_generic(fn):
    """Load Python objects, including msmbuilder Estimators.

    This is a convenience wrapper around Python's ``pickle``
    serialization scheme.

    Parameters
    ----------
    fn : str
        Load this file

    Returns
    -------
    object : object
        The object.
    """
    with open(fn, 'rb') as f:
        return pickle.load(f)


def save_trajs(trajs, fn, meta):
    """Save trajectory-like data

    Data is stored in individual numpy binary files in the
    directory given by ``fn``.

    This method will automatically back up existing files named ``fn``.

    Parameters
    ----------
    trajs : dict of (key, np.ndarray)
        Dictionary of trajectory-like ndarray's keyed on ``meta.index``
        values.
    fn : str
        Where to save the data. This will be a directory containing
        one file per trajectory
    meta : pd.DataFrame
        The DataFrame of metadata
    """
    backup(fn)
    os.mkdir(fn)
    for k in meta.index:
        v = trajs[k]
        np.save(get_fn(fn, k), v)


def load_trajs(fn, meta='meta.pandas.pickl'):
    """Load trajectory-like data

    Data is expected to be stored as if saved by ``save_trajs``.

    This method finds trajectories based on the ``meta`` dataframe.
    If you remove a file (trajectory) from disk, be sure to remove
    its row from the dataframe. If you remove a row from the dataframe,
    be aware that that trajectory (file) will not be loaded, even if
    it exists on disk.

    Parameters
    ----------
    fn : str
        Where the data is saved. This should be a directory containing
        one file per trajectory.
    meta : pd.DataFrame or str
        The DataFrame of metadata. If this is a string, it is interpreted
        as a filename and the dataframe is loaded from disk.

    Returns
    -------
    meta : pd.DataFrame
        The DataFrame of metadata. If you passed in a string (filename)
        to the ``meta`` input, this will be the loaded DataFrame. If
        you gave a DataFrame object, this will just be a reference back
        to that object
    trajs : dict
        Dictionary of trajectory-like np.ndarray's keyed on the values
        of ``meta.index``.
    """
    if isinstance(meta, str):
        meta = load_meta(meta_fn=meta)
    trajs = {}
    for k in meta.index:
        trajs[k] = np.load(get_fn(fn, k))
    return meta, trajs
