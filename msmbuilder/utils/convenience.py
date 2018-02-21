from __future__ import print_function, division, absolute_import

def unique(seq):
    '''Returns a list of unique items maintaining the order of the original.
    '''
    seen = set()
    seen_add = seen.add
    return [x for x in seq if not (x in seen or seen_add(x))]
