#include "utils.h"

#include <cmath>
#include <cstdlib>
#include <map>


int triu_to_tril(double* upper, size_t length, double* lower)
{
    size_t i = 0, j = 0, k = 0, kk = 0;
    /* solve quadratic forumla to make sure tis length is a valid size */
    /* of a condensed distance matrix */
    const size_t n = (1 + sqrt(8*length + 1)) / 2.0;
    if (n*(n-1)/2 != length) {
        /* error! */
        return -1;
    }

    /* iterate through the (i,j) for lower-triangular format */
    for (i = 0; i < n; i++) {
        for (j = 0; j < i; j++) {
            /* location this (i,j) element in upper triangular format */
            kk = n*j - j * (j+1)/2 + (i-1-j);
            lower[k++] = upper[kk];
        }
    }

    return 1;
}


int contigify_ids(int* ids, size_t length)
{
    size_t i = 0;
    int id, new_id;
    std::map<int, int> mapping;

    for (i = 0; i < length; i++) {
        id = ids[i];
        if (mapping.count(id) == 0) {
	    new_id = static_cast<int>(mapping.size());
            mapping[id] = new_id;
        }
        ids[i] = mapping[id];
    }

    return 1;
}
