#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <float.h>
#include <stdio.h>
#include "cephes.h"
#include "spleval.h"
#include "aligned_alloc.h"

//#define DEBUG
#ifdef DEBUG
#define ASSERT_CLOSE(x, y)    if (abs((x)-(y)) > 1e-6) { printf("Assert Failed\nx: %f\ny %f\n", x, y); exit(1); }
#endif



int fitinvkappa(long n_samples, long n_features, long n_components,
                const double* posteriors, const double* obs, const double* means,
                double* out) {
  /*  Implements the following python code in C. There are a few loop
   *  reorderings to try to speed up the cache locality.
   *
   *  for i in range(self.n_features):
   *    for j in range(self.n_components):
   *      numerator = np.sum(posteriors[:, j] * np.cos(obs[:, i] - means_[j, i]))
   *      denominator = np.sum(posteriors[:, j])
   *      inv_kappas[j, i] = numerator / denominator
   *
   */
  long i, j, k;
  double meanshifted, posterior_kj;
  double *num, *denom;

  num = (double*) malloc_simd(n_components * n_features * sizeof(double));
  denom = (double*) malloc_simd(n_components * n_features * sizeof(double));
  if (NULL == num || NULL == denom) {
    fprintf(stderr, "fitinvkappa: Memory allocation failure");
    exit(EXIT_FAILURE);

  }
  memset(num, 0, n_components*n_features*sizeof(double));
  memset(denom, 0, n_components*n_features*sizeof(double));

  for (k = 0; k < n_samples; k++) {
    for (j = 0; j < n_components; j++) {
      posterior_kj = posteriors[k*n_components + j];
      for (i = 0; i < n_features; i++) {
        meanshifted = obs[k*n_features + i] - means[j*n_features + i];
        num[j*n_features + i] += posterior_kj * cos(meanshifted);
        denom[j*n_features + i] += posterior_kj;
      }
    }
  }

  // do the division at the end
  for (i = 0; i < n_features*n_components; i++)
    out[i] = num[i] / denom[i];

  free_simd(num);
  free_simd(denom);
  return 1;
}

int compute_log_likelihood(const double* obs, const double* means,
                           const double* kappas, long n_samples,
                           long n_components, long n_features,
                           double* out) {
  /* Log likelihood of each observation in each state (von Mises distribution)

     Parameters
     ----------
     obs : array, shape=[n_samples, n_components]
     means : array, shape=[n_components, n_features]
     kappas : array, shape=[n_components, n_features]

     Output
     ------
     out : array, shape=[n_samples, n_components]

     Equivalent Python Code
     ----------------------
     >>> from scipy.stats.distributions import vonmises
     >>> n_components = kappas.shape[0]
     >>> value = np.array([np.sum(vonmises.logpdf(obs, kappas[i], means[i]), axis=1) for i in range(n_components)]).T
  */
  unsigned int i, j, k;
  double *kappa_cos_means, *kappa_sin_means;
  double val, log_numerator, cos_obs_kj, sin_obs_kj;
  const double LOG_2PI = 1.8378770664093453;

  // clear the output
  memset(out, 0, n_samples*n_components*sizeof(double));
  // allocate two workspaces
  kappa_cos_means = (double*) malloc_simd(n_components * n_features * sizeof(double));
  kappa_sin_means = (double*) malloc_simd(n_components * n_features * sizeof(double));
  if (NULL == kappa_cos_means || NULL == kappa_sin_means) {
    fprintf(stderr, "compute_log_likelihood: Memory allocation failure");
    exit(EXIT_FAILURE);
  }

  // this sets the likelihood's denominator
  for (i = 0; i < n_components; i++) {
    for (j = 0; j < n_features; j++) {
      val = LOG_2PI + log(i0(kappas[i*n_features + j]));
      for (k = 0; k < n_samples; k++)
        out[k*n_components+i] -= val;
    }
  }

  // We need to calculate cos(obs[k*n_features + j] - means[i*n_features + j])
  // But we want to avoid having a trig function in the inner tripple loop,
  // so we use the double angle formula to split up the computation into cos(x)cos(y) + sin(x)*sin(y)
  // where each of the terms can be computed in a double loop.
  for (i = 0; i < n_components; i++) {
    for (j = 0; j < n_features; j++) {
      kappa_cos_means[j*n_components + i] = kappas[i*n_features + j] * cos(means[i*n_features + j]);
      kappa_sin_means[j*n_components + i] = kappas[i*n_features + j] * sin(means[i*n_features + j]);
    }
  }

  for (k = 0; k < n_samples; k++) {
    for (j = 0; j < n_features; j++) {
      cos_obs_kj = cos(obs[k*n_features + j]);
      sin_obs_kj = sin(obs[k*n_features + j]);
      for (i = 0; i < n_components; i++) {
        log_numerator = (cos_obs_kj*kappa_cos_means[j*n_components + i] + 
        sin_obs_kj*kappa_sin_means[j*n_components + i]);
        out[k*n_components + i] += log_numerator;

        #ifdef DEBUG
          double log_numerator2 = kappas[i*n_features+j]*cos(obs[k*n_features + j] - means[i*n_features + j]);
          ASSERT_CLOSE(log_numerator, log_numerator2);
        #endif
      }
    }
  }

  free_simd(kappa_cos_means);
  free_simd(kappa_sin_means);
  return 1;
}


void inv_mbessel_ratio(double* x, size_t n) {
  // Inverse the function given by the ratio modified Bessel function of the
  // first kind of order 1 to the modified Bessel function of the first kind
  // of order 0.
  //
  // y = A(x) = I_1(x) / I_0(x)
  //
  // This function computes A^(-1)(y) by way of a precomputed spline
  // interpolation. The spline coefficients are calculated at build time
  // by setup.py and saved in .dat files, which are #included here.

  static const double SPLINE_x[] = {
    #include "data/inv_mbessel_x.dat"
  };
  static const double SPLINE_y[] = {
    #include "data/inv_mbessel_y.dat"
  };
  static const double SPLINE_deriv[] = {
    #include "data/inv_mbessel_deriv.dat"
  };

  const size_t n_splinepoints = sizeof(SPLINE_x) / sizeof(SPLINE_x[0]);
  size_t i;
  double t;
  for (i = 0; i < n; i++) {
    t = x[i];

    if (t < SPLINE_x[0])
      t = SPLINE_x[0];
    else if (t > SPLINE_x[n_splinepoints-1])
      t = SPLINE_x[n_splinepoints-1];

    x[i] = exp(evaluateSpline(SPLINE_x, SPLINE_y, SPLINE_deriv, n_splinepoints, t));

  }
}
