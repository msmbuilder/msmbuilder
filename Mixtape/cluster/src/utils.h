#ifndef MIXTAPE_CLUSTER_UTILS_H
#define MIXTAPE_CLUSTER_UTILS_H
#include <map>

/*
Convert a distance matrix storage in upper-triangular format to lower
triangular format.

Upper triangular format stores the rows of the upper triangular portion
of the (symmetric) matrix, (not including the diagonal) contiguously, and
lower triangular format stores the rows of the lower triangular portion
of matrix contiguously.

This function converts from cpndeupper format to lower format.

Example
-------
Upper triangular:

    [x   d(0,1)  d(0,2)  d(0,3)]
    [x     x     d(1,2)  d(1,3)]
    [x     x        x    d(2,3)]
    [x     x        x       x  ]

The condensed upper triangular format (reading the rows across):

    [ d(0,1)  d(0,2)  d(0,3), d(1,2)  d(1,3), d(2,3) ]

Lower triangular:
    [   x      x        x      x]
    [d(1,0)    x        x      x]
    [d(2,0)  d(2,1)     x      x]
    [d(3,0)  d(3,1)    d(3,2)  x]

The condensed lower triangular format (reading the rows across):

    [d(1,0)  d(2,0)  d(2,1), d(3,0)  d(3,1), d(3,2)]
*/
int triu_to_tril(const double* upper, size_t length, double* lower);


/*
Renumber cluster ids to go from 0 to n_clusters - 1.
This function modifies the array inplace.
*/
std::map<int, int> contigify_ids(int* ids, size_t length);

double intertia_euclidean(const double* X, const int* labels,
                          const int* cluster_ids, int n_samples, int n_features);

#endif