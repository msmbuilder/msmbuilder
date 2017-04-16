#include <cfloat>
#include <cmath>
#ifdef _OPENMP
#include <omp.h>
#endif
#include "distance_kernels.h"


void assign_nearest_double(const double* X, const double* Y,
                             const char* metric, const npy_intp* X_indices, npy_intp n_X,
                             npy_intp n_Y, npy_intp n_features, npy_intp n_X_indices,
                             npy_intp* assignments, double *inertia, int n_threads)
{
    double d = 0, min_d = 0; *inertia = 0;
    npy_intp i, j;
    double (*metricfunc) (const double *u, const double *v, npy_intp n) = \
            metric_double(metric);
    if (metricfunc == NULL) {
        fprintf(stderr, "Error");
        return;
    }
#ifdef _OPENMP
    if (n_threads==-1) n_threads=omp_get_num_procs();
    omp_set_num_threads(n_threads);
#endif

    if (X_indices == NULL) {
#ifdef _OPENMP
#pragma omp parallel for \
      shared(inertia,X,n_X,n_Y,assignments,n_features) \
      private(i,min_d,j,d)
#endif
        for (i = 0; i < n_X; i++) {
            min_d = DBL_MAX;
            for (j = 0; j < n_Y; j++) {
                d = metricfunc(&X[i*n_features], &Y[j*n_features], n_features);
                if (d < min_d) {
                    min_d = d;
                    assignments[i] = j;
                }
            }
            *inertia += min_d;
        }
    } else {
#ifdef _OPENMP
#pragma omp parallel for \
      shared(inertia,X,n_X_indices,n_Y,assignments,n_features) \
      private(i,min_d,j,d)
#endif
        for (i = 0; i < n_X_indices; i++) {
            min_d = DBL_MAX;
            for (j = 0; j < n_Y; j++) {
                d = metricfunc(&X[X_indices[i]*n_features], &Y[j*n_features], n_features);
                if (d < min_d) {
                    min_d = d;
                    assignments[i] = j;
                }
            }
            *inertia += min_d;
        }
    }

    return;
}


void assign_nearest_float(const float* X, const float* Y,
                            const char* metric, const npy_intp* X_indices, npy_intp n_X,
                            npy_intp n_Y, npy_intp n_features, npy_intp n_X_indices,
                            npy_intp* assignments, double* inertia, int n_threads)
{
    double d = 0, min_d = 0; *inertia = 0;
    npy_intp i, j;
    double (*metricfunc) (const float *u, const float *v, npy_intp n) = \
            metric_float(metric);
    if (metricfunc == NULL) {
        fprintf(stderr, "Error");
        return;
    }
#ifdef _OPENMP
    if (n_threads==-1) n_threads=omp_get_num_procs();
    omp_set_num_threads(n_threads);
#endif

    if (X_indices == NULL) {
#ifdef _OPENMP
#pragma omp parallel for \
      shared(inertia,X,n_X,n_Y,assignments,n_features) \
      private(i,min_d,j,d)
#endif
        for (i = 0; i < n_X; i++) {
            min_d = DBL_MAX;
            for (j = 0; j < n_Y; j++) {
                d = metricfunc(&X[i*n_features], &Y[j*n_features], n_features);
                if (d < min_d) {
                    min_d = d;
                    assignments[i] = j;
                }
            }
            *inertia += min_d;
        }
    } else {
#ifdef _OPENMP
#pragma omp parallel for \
      shared(inertia,X,n_X_indices,n_Y,assignments,n_features) \
      private(i,min_d,j,d)
#endif
        for (i = 0; i < n_X_indices; i++) {
            min_d = DBL_MAX;
            for (j = 0; j < n_Y; j++) {
                d = metricfunc(&X[X_indices[i]*n_features], &Y[j*n_features], n_features);
                if (d < min_d) {
                    min_d = d;
                    assignments[i] = j;
                }
            }
            *inertia += min_d;
        }
    }

    return;
}
