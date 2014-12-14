#include "distance_kernels.h"


void pdist_double(const double* X, const char* metric, npy_intp n, npy_intp m,
                  double* out)
{
    npy_intp i, j, k;
    const double *u, *v;
    double (*metricfunc) (const double *u, const double *v, npy_intp n) = \
            metric_double(metric);
    if (metricfunc == NULL) {
        fprintf(stderr, "Error");
        return;
    }

    k = 0;
    for (i = 0; i < n; i++) {
        for (j = i+1; j < n; j++) {
            u = X + m * i;
            v = X + m * j;
            out[k++] = metricfunc(u, v, m);
        }
    }
}

void pdist_double_X_indices(const double* X, const char* metric, npy_intp n,
                            npy_intp m, const npy_intp* X_indices,
                            npy_intp n_X_indices, double* out)
{
    npy_intp i, ii, j, jj, k;
    const double *u, *v;
    double (*metricfunc) (const double *u, const double *v, npy_intp n) = \
            metric_double(metric);
    if (metricfunc == NULL) {
        fprintf(stderr, "Error");
        return;
    }

    k = 0;
    for (ii = 0; ii < n_X_indices; ii++) {
        i = X_indices[ii];
        for (jj = ii+1; jj < n_X_indices; jj++) {
            j = X_indices[jj];
            u = X + m * i;
            v = X + m * j;
            out[k++] = metricfunc(u, v, m);
        }
    }
}


void pdist_float(const float* X, const char* metric, npy_intp n, npy_intp m,
                 double* out)
{
    npy_intp i, j, k;
    const float *u, *v;
    double (*metricfunc) (const float *u, const float *v, npy_intp n) = \
            metric_float(metric);
    if (metricfunc == NULL) {
        fprintf(stderr, "Error");
        return;
    }

    k = 0;
    for (i = 0; i < n; i++) {
        for (j = i+1; j < n; j++) {
            u = X + m * i;
            v = X + m * j;
            out[k++] = metricfunc(u, v, m);
        }
    }
}
void pdist_float_X_indices(const float* X, const char* metric, npy_intp n,
                           npy_intp m, const npy_intp* X_indices,
                           npy_intp n_X_indices, double* out)
{
    npy_intp i, ii, j, jj, k;
    const float *u, *v;
    double (*metricfunc) (const float *u, const float *v, npy_intp n) = \
            metric_float(metric);
    if (metricfunc == NULL) {
        fprintf(stderr, "Error");
        return;
    }

    k = 0;
    for (ii = 0; ii < n_X_indices; ii++) {
        i = X_indices[ii];
        for (jj = ii+1; jj < n_X_indices; jj++) {
            j = X_indices[jj];
            u = X + m * i;
            v = X + m * j;
            out[k++] = metricfunc(u, v, m);
        }
    }
}