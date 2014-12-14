#include "distance_kernels.h"


void dist_double(const double* X, const double* y, const char* metric, npy_intp n,
                 npy_intp m, double* out)
{
    npy_intp i;
    const double *u;
    double (*metricfunc) (const double *u, const double *v, npy_intp n) = \
            metric_double(metric);
    if (metricfunc == NULL) {
        fprintf(stderr, "Error");
        return;
    }

    for (i = 0; i < n; i++) {
        u = X + m * i;
        out[i] = metricfunc(u, y, m);
    }
}


void dist_double_X_indices(const double* X, const double* y, const char* metric,
                           npy_intp n, npy_intp m, const npy_intp* X_indices,
                           npy_intp n_X_indices, double* out)
{
    npy_intp i, ii;
    const double *u;
    double (*metricfunc) (const double *u, const double *v, npy_intp n) = \
            metric_double(metric);
    if (metricfunc == NULL) {
        fprintf(stderr, "Error");
        return;
    }

    for (ii = 0; ii < n_X_indices; ii++) {
        i = X_indices[ii];
        u = X + m * i;
        out[ii] = metricfunc(u, y, m);
    }
}


void dist_float(const float* X, const float* y, const char* metric, npy_intp n,
                npy_intp m, double* out)
{
    npy_intp i;
    const float *u;
    double (*metricfunc) (const float *u, const float *v, npy_intp n) = \
            metric_float(metric);
    if (metricfunc == NULL) {
        fprintf(stderr, "Error");
        return;
    }

    for (i = 0; i < n; i++) {
        u = X + m * i;
        out[i] = metricfunc(u, y, m);
    }
}

void dist_float_X_indices(const float* X, const float* y, const char* metric,
                          npy_intp n, npy_intp m, const npy_intp* X_indices,
                          npy_intp n_X_indices, double* out)
{
    npy_intp i, ii;
    const float *u;
    double (*metricfunc) (const float *u, const float *v, npy_intp n) = \
            metric_float(metric);
    if (metricfunc == NULL) {
        fprintf(stderr, "Error");
        return;
    }

    for (ii = 0; ii < n_X_indices; ii++) {
        i = X_indices[ii];
        u = X + m * i;
        out[ii] = metricfunc(u, y, m);
    }
}
