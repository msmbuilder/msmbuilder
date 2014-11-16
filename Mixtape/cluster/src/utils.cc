#include "utils.h"

#include <cmath>
#include <cstdlib>
#include <cstddef>
#include <map>


int triu_to_tril(const double* upper, size_t length, double* lower)
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


std::map<int, int> contigify_ids(int* ids, size_t length)
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

    return mapping;
}


static double
euclidean_distance_double(const double *u, const double *v, size_t n)
{
    double s = 0.0, d;
    size_t i;

    for (i = 0; i < n; i++) {
        d = u[i] - v[i];
        s += d * d;
    }
    return sqrt(s);
}

double intertia_euclidean(const double* X, const int* labels,
                       const int* cluster_ids, int n_samples, int n_features)
{
    size_t i = 0;
    double d = 0;
    for (i = 0; i < n_samples; i++) {
        d += euclidean_distance_double(
            &X[i*n_features], &X[cluster_ids[labels[i]]*n_features], n_features);
    }
    return d;
}