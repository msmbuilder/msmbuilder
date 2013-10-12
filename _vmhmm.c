#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdio.h>

int fitinvkappa(long n_samples, long n_features, long n_components,
                 double* posteriors, double* obs, double* means, double* out) {
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
  int err;
  double meanshifted, posterior_kj;
  double *num, *denom;

  err = posix_memalign((void**) &num, 16, n_components*n_features*sizeof(double));
  err = posix_memalign((void**) &denom, 16, n_components*n_features*sizeof(double));
  if (NULL == num || NULL == denom) {
    fprintf(stderr, "Memory allocation failure");
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

  free(num);
  free(denom);
  return 1;
}