#include "math.h"
#include "stdio.h"
#include "stdlib.h"
#include "gaussian_likelihood.h"

void gaussian_loglikelihood_diag(const float* __restrict__ sequences, const float* __restrict__ means,
                                 const float* __restrict__ variances, const size_t n_trajs,
                                 const size_t* __restrict__ n_observations, const size_t n_states, const size_t n_features,
                                 float* __restrict__ loglikelihoods)
{
    int s, t, i, j;
    float temp;
    static const float log_M_2_PI = 1.8378770664093453f; // np.log(2*np.pi)
    const float* __restrict__ _sequences = sequences;
    float* __restrict__ _loglikelihoods = loglikelihoods;


    // These functions of the means and variances can be factored out of the inner loop
    // for speed
    float* __restrict__ means_over_vars = (float*) malloc(n_states*n_features*sizeof(float));
    float* __restrict__ means2_over_vars = (float*) malloc(n_states*n_features*sizeof(float));
    float* __restrict__ log_vars = (float*) malloc(n_states*n_features*sizeof(float));
    if (means2_over_vars == NULL || means2_over_vars == NULL || log_vars == NULL) {
        fprintf(stderr, "Memory allocation failure");
        exit(EXIT_FAILURE);
    }
    for (i = 0; i < n_states*n_features; i++) {
        means_over_vars[i] = means[i] / variances[i];
        means2_over_vars[i] = means_over_vars[i]*means[i];
        log_vars[i] = log(variances[i]);
    }


    for (s = 0; s < n_trajs; s++) {
        for (t = 0; t < n_observations[s]; t++) {
            for (j = 0; j < n_states; j++) {
                temp = 0.0f;
                for (i = 0; i < n_features; i++)
                    temp += means2_over_vars[j*n_features + i]          \
                            - 2.0 * _sequences[t*n_features + i]*means_over_vars[j*n_features + i] \
                            + _sequences[t*n_features + i]*_sequences[t*n_features + i] / variances[j*n_features + i] \
                            + log_vars[j*n_features + i];
                
                _loglikelihoods[t*n_states + j] = -0.5 * (n_features * log_M_2_PI + temp);
            }
        }
        _loglikelihoods += n_observations[s]*n_states;
        _sequences += n_observations[s]*n_features;
    }

    free(means_over_vars);
    free(means2_over_vars);
    free(log_vars);
}

