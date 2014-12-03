from __future__ import print_function, division, absolute_import
import numpy as np
import mdtraj as md

__all__ = ['list_of_1d', 'check_iter_of_sequences']


def list_of_1d(y):
    if not hasattr(y, '__iter__') or len(y) == 0:
        raise ValueError('Bad input shape')
    if not hasattr(y[0], '__iter__'):
        return [np.array(y)]

    result = []
    for i, x in enumerate(y):
        value = np.array(x)
        if value.ndim != 1:
            raise ValueError(
                "Bad input shape. Element %d has shape %s, but "
                "should be 1D" % (i, str(value.shape)))
        result.append(value)
    return result


def check_iter_of_sequences(sequences, allow_trajectory=False, ndim=2,
                            max_iter=None):
    """Check that ``sequences`` is a iterable of trajectory-like sequences,
    suitable as input to ``fit()`` for estimators following the Mixtape
    API.

    Parameters
    ----------
    sequences : object
        The object to check
    allow_trajectory : bool
        Are ``md.Trajectory``s allowed?
    ndim : int
        The expected dimensionality of the sequences
    max_iter : int, optional
        Only check at maximum the first ``max_iter`` entries in ``sequences``.
    """
    value = True
    for i, X in enumerate(sequences):
        if not isinstance(X, np.ndarray):
            if (not allow_trajectory) and isinstance(X, md.Trajectory):
                value = False
                break
        if not isinstance(X, md.Trajectory) and X.ndim != ndim:
            value = False
            break
        if max_iter is not None and i >= max_iter:
            break

    if not value:
        raise ValueError('sequences must be a list of sequences')
