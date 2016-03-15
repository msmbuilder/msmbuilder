#include "distance_kernels.h"


void cdist_double(const double* XA, const double* XB, const char* metric,
                  npy_intp na, npy_intp nb, npy_intp m, double* out)

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
    for (i = 0; i < na; i++) {
        for (j = 0; j < nb; j++) {
            u = XA + m * i;
            v = XB + m * j;
            out[k++] = metricfunc(u, v, m);
        }
    }
}


void cdist_float(const float* XA, const float* XB, const char* metric,
                  npy_intp na, npy_intp nb, npy_intp m, double* out)

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
    for (i = 0; i < na; i++) {
        for (j = 0; j < nb; j++) {
            u = XA + m * i;
            v = XB + m * j;
            out[k++] = metricfunc(u, v, m);
        }
    }
}