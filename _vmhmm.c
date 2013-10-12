// Copyright (C) 2013  Stanford University
//
// This library is free software; you can redistribute it and/or
// modify it under the terms of the GNU Lesser General Public
// License as published by the Free Software Foundation; either
// version 2.1 of the License, or (at your option) any later version.
//
// This library is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
// Lesser General Public License for more details.
//
// You should have received a copy of the GNU Lesser General Public
// License along with this library; if not, write to the Free Software
// Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301  USA

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
  int err;
  long i, j, k;
  double meanshifted, posterior_kj;
  double *num, *denom;

  err = posix_memalign((void**) &num, 16, n_components * n_features * sizeof(double));
  err = posix_memalign((void**) &denom, 16, n_components * n_features * sizeof(double));
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
