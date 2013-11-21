#include "math.h"
#include "stdio.h"
#include "stdlib.h"
#include "gaussian_likelihood.h"

void gaussian_loglikelihood_diag(const float* __restrict__ sequence,
                                 const float* __restrict__ sequence2,
                                 const float* __restrict__ means,
                                 const float* __restrict__ variances,
                                 const float* __restrict__ means_over_variances,
                                 const float* __restrict__ means2_over_variances,
                                 const float* __restrict__ log_variances,
                                 const int n_observations,
                                 const int n_states, const int n_features,
                                 float* __restrict__ loglikelihoods)
{
    int t, i, j;
    float temp;
    static const float log_M_2_PI = 1.8378770664093453f; // np.log(2*np.pi)

    for (t = 0; t < n_observations; t++) {
        for (j = 0; j < n_states; j++) {
            temp = 0.0f;
            for (i = 0; i < n_features; i++) {
                temp += means2_over_variances[j*n_features + i]              \
                        - 2.0 * sequence[t*n_features + i]*means_over_variances[j*n_features + i] \
                        + sequence2[t*n_features + i] / variances[j*n_features + i] \
                        + log_variances[j*n_features + i];
            }
            loglikelihoods[t*n_states + j] = -0.5 * (n_features * log_M_2_PI + temp);
        }
    }
}
