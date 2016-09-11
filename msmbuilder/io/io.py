# Author: Matthew Harrigan <matthew.harrigan@outlook.com>
# Contributors:
# Copyright (c) 2016, Stanford University
# All rights reserved.

from __future__ import print_function, division, absolute_import

import os
import pickle
import re
import shutil
import stat
import warnings

import mdtraj as md
import numpy as np
import pandas as pd
from jinja2 import Environment, PackageLoader

__all__ = ['backup', 'preload_top', 'preload_tops', 'load_meta', 'load_generic',
           'load_trajs', 'save_meta', 'render_meta', 'save_generic',
           'itertrajs', 'save_trajs', 'ProjectTemplatej']


class BackupWarning(UserWarning):
    pass


def backup(fn):
    """If ``fn`` exists, rename it and issue a warning

    This function will rename an existing filename {fn}.bak.{i} where
    i is the smallest integer that gives a filename that doesn't exist.
    This naively uses a while loop to find such a filename, so there
    shouldn't be too many existing backups or performance will degrade.

    Parameters
    ----------
    fn : str
        The filename to check.
    """
    if not os.path.exists(fn):
        return

    backnum = 1
    backfmt = "{fn}.bak.{backnum}"
    trial_fn = backfmt.format(fn=fn, backnum=backnum)
    while os.path.exists(trial_fn):
        backnum += 1
        trial_fn = backfmt.format(fn=fn, backnum=backnum)

    warnings.warn("{fn} exists. Moving it to {newfn}"
                  .format(fn=fn, newfn=trial_fn),
                  BackupWarning)
    shutil.move(fn, trial_fn)


def chmod_plus_x(fn):
    st = os.stat(fn)
    os.chmod(fn, st.st_mode | stat.S_IEXEC)


def default_key_to_path(key, dfmt="{}", ffmt="{}.npy"):
    """Turn an arbitrary python object into a filename

    This uses string formatting, so make sure your keys map
    to unique strings. If the key is a tuple, it will join each
    element of the tuple with '/', resulting in a filesystem
    hierarchy of files.
    """
    if isinstance(key, tuple):
        paths = [dfmt.format(k) for k in key[:-1]]
        paths += [ffmt.format(key[-1])]
        return os.path.join(*paths)
    else:
        return ffmt.format(key)


def validate_keys(keys, key_to_path_func=None,
                  valid_re=r"[a-zA-Z0-9_\-\.]+(\/[a-zA-Z0-9_\-\.]+)*"):
    if key_to_path_func is None:
        key_to_path_func = default_key_to_path

    err = "Key must match regular expression {}".format(valid_re)
    for k in keys:
        ks = key_to_path_func(k)
        assert isinstance(ks, str), "Key must convert to a string"
        assert re.match(valid_re, ks), err


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
        tops[tfn] = md.load_topology(tfn)
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


def itertrajs(meta, stride=1):
    """Load one mdtraj trajectory at a time and yield it.

    MDTraj does striding badly. It reads in the whole trajectory and
    then performs a stride. We join(iterload) to conserve memory.
    """

    tops = preload_tops(meta)
    for i, row in meta.iterrows():
        yield i, md.join(md.iterload(row['traj_fn'],
                                     top=tops[row['top_fn']],
                                     stride=stride),
                         discard_overlapping_frames=False,
                         check_topology=False)


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
    backup(meta_fn)
    pd.to_pickle(meta, meta_fn)


def render_meta(meta, fn="meta.pandas.html",
                title="Project Metadata - MSMBuilder", pandas_kwargs=None):
    """Render a metadata dataframe as an html webpage for inspection.

    Parameters
    ----------
    meta : pd.Dataframe
        The DataFrame of metadata
    fn : str
        Output filename (should end in html)
    title : str
        Page title
    pandas_kwargs : dict
        Arguments to be passed to pandas

    """
    if pandas_kwargs is None:
        pandas_kwargs = {}

    kwargs_with_defaults = {
        'classes': ('table', 'table-condensed', 'table-hover'),
    }
    kwargs_with_defaults.update(**pandas_kwargs)

    env = Environment(loader=PackageLoader('msmbuilder', 'io_templates'))
    templ = env.get_template("twitter-bootstrap.html")
    rendered = templ.render(
        title=title,
        content=meta.to_html(**kwargs_with_defaults)
    )

    # Ugh, pandas hardcodes border="1"
    rendered = re.sub(r' border="1"', '', rendered)

    backup(fn)
    with open(fn, 'w') as f:
        f.write(rendered)


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


def save_trajs(trajs, fn, meta, key_to_path=None):
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
    if key_to_path is None:
        key_to_path = default_key_to_path

    validate_keys(meta.index, key_to_path)
    backup(fn)
    os.mkdir(fn)
    for k in meta.index:
        v = trajs[k]
        npy_fn = os.path.join(fn, key_to_path(k))
        os.makedirs(os.path.dirname(npy_fn), exist_ok=True)
        np.save(npy_fn, v)


def load_trajs(fn, meta='meta.pandas.pickl', key_to_path=None):
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
    if key_to_path is None:
        key_to_path = default_key_to_path

    if isinstance(meta, str):
        meta = load_meta(meta_fn=meta)
    trajs = {}
    for k in meta.index:
        trajs[k] = np.load(os.path.join(fn, key_to_path(k)))
    return meta, trajs
