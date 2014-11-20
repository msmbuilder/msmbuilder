#include "distance_kernels.h"

double sumdist_double(const double* X, const char* metric, npy_intp n, npy_intp m,
                      const npy_intp* pairs, npy_intp p)
{
    npy_intp i;
    double s = 0;
    const double *u, *v;
    double (*metricfunc) (const double *u, const double *v, npy_intp n) = \
            metric_double(metric);
    if (metricfunc == NULL) {
        fprintf(stderr, "Error");
        return -1;
    }

    for (i = 0; i < p; i++) {
        u = X + m * pairs[2*i];
        v = X + m * pairs[2*i+1];
        s += metricfunc(u, v, m);
    }

    return s;
}


double sumdist_float(const float* X, const char* metric, npy_intp n, npy_intp m,
                     const npy_intp* pairs, npy_intp p)
{
    npy_intp i;
    double s = 0;
    const float *u, *v;
    double (*metricfunc) (const float *u, const float *v, npy_intp n) = \
            metric_float(metric);
    if (metricfunc == NULL) {
        fprintf(stderr, "Error");
        return -1;
    }
    for (i = 0; i < p; i++) {
        u = X + m * pairs[2*i];
        v = X + m * pairs[2*i+1];
        s += metricfunc(u, v, m);
    }

    return s;
}
