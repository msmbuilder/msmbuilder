/**
 * Author: Damian Eads
 * Date:   September 22, 2007 (moved to new file on June 8, 2008)
 *
 * Copyright (c) 2007, 2008, Damian Eads. All rights reserved.
 * Adapted for incorporation into Scipy, April 9, 2008.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions
 * are met:
 *   - Redistributions of source code must retain the above
 *     copyright notice, this list of conditions and the
 *     following disclaimer.
 *   - Redistributions in binary form must reproduce the above copyright
 *     notice, this list of conditions and the following disclaimer
 *     in the documentation and/or other materials provided with the
 *     distribution.
 *   - Neither the name of the author nor the names of its
 *     contributors may be used to endorse or promote products derived
 *     from this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
 * "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
 * LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
 * A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
 * OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
 * SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
 * LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
 * DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
 * THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

#ifndef MIXTAPE_LIBDISTANCE_KERNELS_H
#define MIXTAPE_LIBDISTANCE_KERNELS_H
#ifdef __cplusplus
extern "C" {
#endif

static NPY_INLINE double
sqeuclidean_distance_double(const double *u, const double *v, npy_intp n)
{
    double s = 0.0, d;
    npy_intp i;

    for (i = 0; i < n; i++) {
        d = u[i] - v[i];
        s += d * d;
    }
    return s;
}

static NPY_INLINE double
sqeuclidean_distance_float(const float *u, const float *v, npy_intp n)
{
    double s = 0.0, d;
    npy_intp i;

    for (i = 0; i < n; i++) {
        d = u[i] - v[i];
        s += d * d;
    }
    return s;
}

static NPY_INLINE double
euclidean_distance_double(const double *u, const double *v, npy_intp n)
{
    return sqrt(sqeuclidean_distance_double(u, v, n));
}

static NPY_INLINE double
euclidean_distance_float(const float *u, const float *v, npy_intp n)
{
    return sqrt(sqeuclidean_distance_float(u, v, n));
}

static NPY_INLINE double
chebyshev_distance_double(const double *u, const double *v, npy_intp n)
{
    double d, maxv = 0.0;
    npy_intp i;

    for (i = 0; i < n; i++) {
        d = fabs(u[i] - v[i]);
        if (d > maxv) {
            maxv = d;
        }
    }
    return maxv;
}

static NPY_INLINE double
chebyshev_distance_float(const float *u, const float *v, npy_intp n)
{
    double d, maxv = 0.0;
    npy_intp i;

    for (i = 0; i < n; i++) {
        d = fabs(u[i] - v[i]);
        if (d > maxv) {
            maxv = d;
        }
    }
    return maxv;
}

static NPY_INLINE double
canberra_distance_double(const double *u, const double *v, npy_intp n)
{
    double snum = 0.0, sdenom = 0.0, tot = 0.0;
    npy_intp i;

    for (i = 0; i < n; i++) {
        snum = fabs(u[i] - v[i]);
        sdenom = fabs(u[i]) + fabs(v[i]);
        if (sdenom > 0.0) {
            tot += snum / sdenom;
        }
    }
    return tot;
}

static NPY_INLINE double
canberra_distance_float(const float *u, const float *v, npy_intp n)
{
    double snum = 0.0, sdenom = 0.0, tot = 0.0;
    npy_intp i;

    for (i = 0; i < n; i++) {
        snum = fabs(u[i] - v[i]);
        sdenom = fabs(u[i]) + fabs(v[i]);
        if (sdenom > 0.0) {
            tot += snum / sdenom;
        }
    }
    return tot;
}

static NPY_INLINE double
bray_curtis_distance_double(const double *u, const double *v, npy_intp n)
{
    double s1 = 0.0, s2 = 0.0;
    npy_intp i;

    for (i = 0; i < n; i++) {
        s1 += fabs(u[i] - v[i]);
        s2 += fabs(u[i] + v[i]);
    }
    return s1 / s2;
}

static NPY_INLINE double
bray_curtis_distance_float(const float *u, const float *v, npy_intp n)
{
    double s1 = 0.0, s2 = 0.0;
    npy_intp i;

    for (i = 0; i < n; i++) {
        s1 += fabs(u[i] - v[i]);
        s2 += fabs(u[i] + v[i]);
    }
    return s1 / s2;
}

static NPY_INLINE double
hamming_distance_double(const double *u, const double *v, npy_intp n)
{
    double s = 0.0;
    npy_intp i;

    for (i = 0; i < n; i++) {
        s += (u[i] != v[i]);
    }
    return s / n;
}

static NPY_INLINE double
hamming_distance_float(const float *u, const float *v, npy_intp n)
{
    double s = 0.0;
    npy_intp i;

    for (i = 0; i < n; i++) {
        s += (u[i] != v[i]);
    }
    return s / n;
}

static NPY_INLINE double
jaccard_distance_double(const double *u, const double *v, npy_intp n)
{
    double denom = 0.0, num = 0.0;
    npy_intp i;

    for (i = 0; i < n; i++) {
        num += (u[i] != v[i]) & ((u[i] != 0.0) | (v[i] != 0.0));
        denom += (u[i] != 0.0) | (v[i] != 0.0);
    }
    return num / denom;
}

static NPY_INLINE double
jaccard_distance_float(const float *u, const float *v, npy_intp n)
{
    double denom = 0.0, num = 0.0;
    npy_intp i;

    for (i = 0; i < n; i++) {
        num += (u[i] != v[i]) & ((u[i] != 0.0) | (v[i] != 0.0));
        denom += (u[i] != 0.0) | (v[i] != 0.0);
    }
    return num / denom;
}


static NPY_INLINE double
city_block_distance_double(const double *u, const double *v, npy_intp n)
{
    double s = 0.0, d;
    npy_intp i;

    for (i = 0; i < n; i++) {
        d = fabs(u[i] - v[i]);
        s = s + d;
    }
    return s;
}

static NPY_INLINE double
city_block_distance_float(const float *u, const float *v, npy_intp n)
{
    double s = 0.0, d;
    npy_intp i;

    for (i = 0; i < n; i++) {
        d = fabs(u[i] - v[i]);
        s = s + d;
    }
    return s;
}


static double(*metric_double(const char* metric))
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

static double(*metric_float(const char* metric))
  (const float *u, const float *v, npy_intp n)
{
    if (strcmp(metric, "euclidean") == 0) {
        return &euclidean_distance_float;  
    } else if (strcmp(metric, "sqeuclidean") == 0) {
        return &sqeuclidean_distance_float;
    } else if (strcmp(metric, "cityblock") == 0) {
      return &city_block_distance_float;
    } else if (strcmp(metric, "chebyshev") == 0) {
      return &chebyshev_distance_float;
    } else if (strcmp(metric, "canberra") == 0) {
      return &canberra_distance_float;
    } else if (strcmp(metric, "braycurtis") == 0) {
      return &bray_curtis_distance_float;
    } else if (strcmp(metric, "hamming") == 0) {
      return &hamming_distance_float;
    } else if (strcmp(metric, "jaccard") == 0) {
      return &jaccard_distance_float;
    } else if (strcmp(metric, "cityblock") == 0) {
      return &city_block_distance_float;
    }
    return NULL;
}



#ifdef __cplusplus
}
#endif
#endif