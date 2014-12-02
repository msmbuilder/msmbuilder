"""Binary search algorithm
"""
from numpy cimport npy_intp


cpdef inline npy_intp bsearch(const npy_intp[::1] haystack, npy_intp needle) nogil:
    """Return the index of `needle` in a sorted list, `haystack`. If not
    found, return -1.

    Parameters
    ----------
    haystack : array of ints
        Array. This array must be in sorted order.
    needle : int
        The element to search for

    Returns
    -------
    index : int
        The index of `needle` in `haystack`, between 0 and `len(haystack)-1`,
        or -1 if needle is not an element in `haystack`.
    """
    # Code adapted from http://rosettacode.org/wiki/Binary_search
    cdef npy_intp mid
    cdef npy_intp low = 0
    cdef npy_intp high = haystack.shape[0] - 1

    while low <= high:
        mid = low + (high - low) / 2  # avoid overflow in `(left + right)/2`
        if haystack[mid] > needle:
            high = mid - 1
        elif haystack[mid] < needle:
            low = mid + 1
        else:
            return mid

    return -1
