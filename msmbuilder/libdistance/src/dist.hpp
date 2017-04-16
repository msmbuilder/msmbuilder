#include "distance_kernels.h"
#ifdef _OPENMP
#include <omp.h>
#endif

void dist_double(const double* X, const double* y, const char* metric, npy_intp n,
                 npy_intp m, double* out, int n_threads)
{
    npy_intp i;
    const double *u;
    double (*metricfunc) (const double *u, const double *v, npy_intp n) = \
            metric_double(metric);
    if (metricfunc == NULL) {
        fprintf(stderr, "Error");
        return;
    }

#ifdef _OPENMP
    if (n_threads==-1) n_threads=omp_get_num_procs();
    omp_set_num_threads(n_threads);
#pragma omp parallel for \
        shared(X,y,m,n,out) private(i,u)
#endif
    for (i = 0; i < n; i++) {
        u = X + m * i;
        out[i] = metricfunc(u, y, m);
    }
}


void dist_double_X_indices(const double* X, const double* y, const char* metric,
                           npy_intp n, npy_intp m, const npy_intp* X_indices,
                           npy_intp n_X_indices, double* out, int n_threads)
{
    npy_intp i, ii;
    const double *u;
    double (*metricfunc) (const double *u, const double *v, npy_intp n) = \
            metric_double(metric);
    if (metricfunc == NULL) {
        fprintf(stderr, "Error");
        return;
    }
    
#ifdef _OPENMP
    if (n_threads==-1) n_threads=omp_get_num_procs();
    omp_set_num_threads(n_threads);
#pragma omp parallel for \
        shared(X,X_indices,n_X_indices,y,m,out) private(ii,i,u)
#endif
    for (ii = 0; ii < n_X_indices; ii++) {
        i = X_indices[ii];
        u = X + m * i;
        out[ii] = metricfunc(u, y, m);
    }
}


void dist_float(const float* X, const float* y, const char* metric, npy_intp n,
                npy_intp m, double* out, int n_threads)
{
    npy_intp i;
    const float *u;
    double (*metricfunc) (const float *u, const float *v, npy_intp n) = \
            metric_float(metric);
    if (metricfunc == NULL) {
        fprintf(stderr, "Error");
        return;
    }

#ifdef _OPENMP
    if (n_threads==-1) n_threads=omp_get_num_procs();
    omp_set_num_threads(n_threads);
#pragma omp parallel for \
        shared(X,y,m,n,out) private(i,u)
#endif
    for (i = 0; i < n; i++) {
        u = X + m * i;
        out[i] = metricfunc(u, y, m);
    }
}

void dist_float_X_indices(const float* X, const float* y, const char* metric,
                          npy_intp n, npy_intp m, const npy_intp* X_indices,
                          npy_intp n_X_indices, double* out, int n_threads)
{
    npy_intp i, ii;
    const float *u;
    double (*metricfunc) (const float *u, const float *v, npy_intp n) = \
            metric_float(metric);
    if (metricfunc == NULL) {
        fprintf(stderr, "Error");
        return;
    }
    
#ifdef _OPENMP
    if (n_threads==-1) n_threads=omp_get_num_procs();
    omp_set_num_threads(n_threads);
#pragma omp parallel for \
        shared(X,X_indices,n_X_indices,y,m,out) private(ii,i,u)
#endif
    for (ii = 0; ii < n_X_indices; ii++) {
        i = X_indices[ii];
        u = X + m * i;
        out[ii] = metricfunc(u, y, m);
    }
}
