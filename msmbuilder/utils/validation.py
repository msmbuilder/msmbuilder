from __future__ import print_function, division, absolute_import
import numpy as np
import mdtraj as md

__all__ = ['list_of_1d', 'check_iter_of_sequences', 'array2d', "check_arrays"]


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
    suitable as input to ``fit()`` for estimators following the MSMBuilder
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


def array2d(X, dtype=None, order=None, copy=False, force_all_finite=True):
    """Returns at least 2-d array with data from X"""
    X_2d = np.asarray(np.atleast_2d(X), dtype=dtype, order=order)
    if force_all_finite:
        _assert_all_finite(X_2d)
    if X is X_2d and copy:
        X_2d = _safe_copy(X_2d)
    return X_2d


def _assert_all_finite(X):
    """Like assert_all_finite, but only for ndarray."""
    X = np.asanyarray(X)
    if (X.dtype.char in np.typecodes['AllFloat'] and not np.isfinite(X.sum())
            and not np.isfinite(X).all()):
        raise ValueError("Input contains NaN, infinity"
                         " or a value too large for %r." % X.dtype)

def _safe_copy(X):
    # Copy, but keep the order
    return np.copy(X, order='K')


def check_arrays(*arrays, **options):
    """Check that all arrays have consistent first dimensions.

    Checks whether all objects in arrays have the same shape or length.
    By default lists and tuples are converted to numpy arrays.

    It is possible to enforce certain properties, such as dtype, continguity
    and sparse matrix format (if a sparse matrix is passed).

    Converting lists to arrays can be disabled by setting ``allow_lists=True``.
    Lists can then contain arbitrary objects and are not checked for dtype,
    finiteness or anything else but length. Arrays are still checked
    and possibly converted.


    Parameters
    ----------
    *arrays : sequence of arrays or scipy.sparse matrices with same shape[0]
        Python lists or tuples occurring in arrays are converted to 1D numpy
        arrays, unless allow_lists is specified.

    sparse_format : 'csr' | 'csc' | 'dense' | list,  None by default.
        If not None, scipy.sparse matrices will be converted to the format(s)
        specified, or left alone if they are already an accepted format.
        If multiple sparse formats are accepted, pass them in via a list.
        This list can contain 'csc' and/or 'csr.'
        If 'dense', an error is raised when a sparse array is
        passed.

    copy : boolean, False by default
        If copy is True, ensure that returned arrays are copies of the original
        (if not already converted to another format earlier in the process).

    check_ccontiguous : boolean, False by default
        Check that the arrays are C contiguous

    dtype : a numpy dtype instance, None by default
        Enforce a specific dtype.

    allow_lists : bool
        Allow lists of arbitrary objects as input, just check their length.
        Disables

    allow_nans : boolean, False by default
        Allows nans in the arrays
    """
    sparse_format = options.pop('sparse_format', None)
    _validate_sparse_format_options(sparse_format)

    copy = options.pop('copy', False)
    check_ccontiguous = options.pop('check_ccontiguous', False)
    dtype = options.pop('dtype', None)
    allow_lists = options.pop('allow_lists', False)
    allow_nans = options.pop('allow_nans', False)

    if options:
        raise TypeError("Unexpected keyword arguments: %r" % options.keys())

    if len(arrays) == 0:
        return None

    n_samples = _num_samples(arrays[0])

    checked_arrays = []
    for array in arrays:
        array_orig = array
        if array is None:
            # special case: ignore optional y=None kwarg pattern
            checked_arrays.append(array)
            continue
        size = _num_samples(array)

        if size != n_samples:
            raise ValueError("Found array with dim %d. Expected %d"
                             % (size, n_samples))

        if not allow_lists or hasattr(array, "shape"):
            if sp.issparse(array):
                array = _check_sparse_format(array, sparse_format)
                if check_ccontiguous:
                    array.data = np.ascontiguousarray(array.data, dtype=dtype)
                else:
                    array.data = np.asarray(array.data, dtype=dtype)
                if not allow_nans:
                    _assert_all_finite(array.data)
            else:
                if check_ccontiguous:
                    array = np.ascontiguousarray(array, dtype=dtype)
                else:
                    array = np.asarray(array, dtype=dtype)
                if not allow_nans:
                    _assert_all_finite(array)

            if array.ndim >= 3:
                raise ValueError("Found array with dim %d. Expected <= 2" %
                                 array.ndim)

        if copy and array is array_orig:
            array = array.copy()
        checked_arrays.append(array)

    return checked_arrays


def _validate_sparse_format_options(sparse_format):
    """Validates sparse_format options for `check_arrays`"""
    valid_sparse_formats = ('csr', 'csc')
    valid_sparse_options = ('csr', 'csc', 'dense', None)

    is_valid_format = sparse_format in valid_sparse_options
    contains_valid_formats = (
        hasattr(sparse_format, '__iter__') and
        all([f in valid_sparse_formats for f in sparse_format])
    )

    # Validate sparse_format option
    if not(is_valid_format or contains_valid_formats):
        raise ValueError('Unexpected sparse format(s): %r' % sparse_format)



def _num_samples(x):
    """Return number of samples in array-like x."""
    if not hasattr(x, '__len__') and not hasattr(x, 'shape'):
        raise TypeError("Expected sequence or array-like, got %r" % x)
    return x.shape[0] if hasattr(x, 'shape') else len(x)
