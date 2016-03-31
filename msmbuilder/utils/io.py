from __future__ import print_function, division, absolute_import

import contextlib
import pickle
import warnings

import numpy as np
from sklearn.externals.joblib import load as jl_load

__all__ = ['printoptions', 'verbosedump', 'verboseload', 'dump', 'load']

warnings.warn("This module might be deprecated in favor of msmbuilder.io",
              PendingDeprecationWarning)


@contextlib.contextmanager
def printoptions(*args, **kwargs):
    original = np.get_printoptions()
    np.set_printoptions(*args, **kwargs)
    yield
    np.set_printoptions(**original)


def dump(value, filename, compress=None, cache_size=None):
    """Save an arbitrary python object using pickle.

    Parameters
    -----------
    value : any Python object
        The object to store to disk using pickle.
    filename : string
        The name of the file in which it is to be stored
    compress : None
        No longer used
    cache_size : positive number, optional
        No longer used

    See Also
    --------
    load : corresponding loader
    """
    if compress is not None or cache_size is not None:
        warnings.warn("compress and cache_size are no longer valid options")

    with open(filename, 'wb') as f:
        pickle.dump(value, f)


def load(filename):
    """Load an object that has been saved with dump.

    We try to open it using the pickle protocol. As a fallback, we
    use joblib.load. Joblib was the default prior to msmbuilder v3.2

    Parameters
    ----------
    filename : string
        The name of the file to load.
    """
    try:
        with open(filename, 'rb') as f:
            return pickle.load(f)
    except Exception as e1:
        try:
            return jl_load(filename)
        except Exception as e2:
            raise IOError(
                "Unable to load {} using the pickle or joblib protocol.\n"
                "Pickle: {}\n"
                "Joblib: {}".format(filename, e1, e2)
            )


def verbosedump(value, fn, compress=None):
    """Verbose wrapper around dump"""
    print('Saving "%s"... (%s)' % (fn, type(value)))
    dump(value, fn, compress=compress)


def verboseload(fn):
    """Verbose wrapper around load.

    Try to use pickle. If that fails, try to use joblib.
    """
    print('loading "%s"...' % fn)
    return load(fn)
