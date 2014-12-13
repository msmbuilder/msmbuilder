import hashlib
import itertools
import numpy as np
from six.moves import xrange


def iterate_tracker(maxiter, max_nc, verbose=False):
    """Generator that breaks after maxiter, or after the same
    array has been sent in more max_nc times in a row.
    """
    last_hash = None
    last_hash_count = 0
    arr = yield

    for i in xrange(maxiter):
        arr = yield i
        if arr is not None:    
            hsh = hashlib.sha1(arr.view(np.uint8)).hexdigest()
            if last_hash == hsh:
                last_hash_count += 1
            else:
                last_hash = hsh
                last_hash_count = 1

            if last_hash_count >= max_nc:
                if verbose:
                    print('Termination. Over %d iterations without '
                          'change.' % max_nc)
                break