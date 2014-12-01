from __future__ import print_function, division, absolute_import
import contextlib
import numpy as np
from sklearn.externals.joblib import load
from skelearn.externals.joblib import dump as _dump

__all__ = ['printoptions', 'verbosedump', 'verboseload', 'dump', 'load']


@contextlib.contextmanager
def printoptions(*args, **kwargs):
    original = np.get_printoptions()
    np.set_printoptions(*args, **kwargs)
    yield
    np.set_printoptions(**original)


def dump(value, filename, compress=1, cache_size=100):
    """Fast persistence of an arbitrary Python object into a files, with
    dedicated storage for numpy arrays.

    Parameters
    -----------
    value: any Python object
        The object to store to disk
    filename: string
        The name of the file in which it is to be stored
    compress: integer for 0 to 9, optional
        Optional compression level for the data. 0 is no compression.
        Higher means more compression, but also slower read and
        write times. Using a value of 3 is often a good compromise.
        See the notes for more details.
    cache_size: positive number, optional
        Fixes the order of magnitude (in megabytes) of the cache used
        for in-memory compression. Note that this is just an order of
        magnitude estimate and that for big arrays, the code will go
        over this value at dump and at load time.

    Returns
    -------
    filenames: list of strings
        The list of file names in which the data is stored. If
        compress is false, each array is stored in a different file.

    See Also
    --------
    load : corresponding loader
    """
    _dump(value, filename, compress=compress, cache_size=cache_size)


def verbosedump(value, fn, compress=1):
    """verbose wrapper around dump"""
    print('Saving "%s"... (%s)' % (fn, type(value)))
    dump(value, fn, compress=compress)


def verboseload(fn):
    """verbose wrapper around joblib.load"""
    print('loading "%s"...' % fn)
    return load(fn)
