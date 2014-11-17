#include "math.h"
#include "stdio.h"
#include "string.h"
#include "Python.h"
#include "numpy/npy_common.h"
#include "distance_impl.h"
#include "inertia.h"

static double(*setmetric(char* metric))
  (const double *u, const double *v, npy_intp n)
{
    if (strcmp(metric, "euclidean") == 0) {
        return &euclidean_distance_double;  
    } else if (strcmp(metric, "sqeuclidean") == 0) {
        return &sqeuclidean_distance_double;
    } else if (strcmp(metric, "cityblock") == 0) {
      return &city_block_distance_double;
    } else if (strcmp(metric, "chebyshev") == 0) {
      return &chebyshev_distance_double;
    } else if (strcmp(metric, "canberra") == 0) {
      return &canberra_distance_double;
    } else if (strcmp(metric, "braycurtis") == 0) {
      return &bray_curtis_distance_double;
    } else if (strcmp(metric, "hamming") == 0) {
      return &hamming_distance_double;
    } else if (strcmp(metric, "jaccard") == 0) {
      return &jaccard_distance_double;
    } else if (strcmp(metric, "cityblock") == 0) {
      return &city_block_distance_double;
    }
    return NULL;
}


double inertia(const double* X, const int* labels,
                const int* cluster_ids, int n_samples, int n_features,
                const char* metric)
{
    size_t i = 0;
    double d = 0;
    double (*metricfunc) (const double *u, const double *v, npy_intp n) = setmetric(metric);
    if (metricfunc == NULL) {
        fprintf(stderr, "Error");
        return -1;
    }

    for (i = 0; i < n_samples; i++) {
        d += metricfunc(&X[i*n_features], &X[cluster_ids[labels[i]]*n_features],
                        n_features);
    }
    return d;
}