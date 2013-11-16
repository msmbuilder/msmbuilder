#include <stdlib.h>
#include "gaussian_likelihood.h"
#include "assertions.h"

int main() {
    int s, t, i, j;
    const size_t n_states = 4;
    const size_t n_observations[] = {2, 3, 4};
    const int n_trajs = sizeof(n_observations) / sizeof(n_observations[0]);
    const int n_features = 2;
    int n_total_obs = 0;
    for (i = 0; i < n_trajs; i++)
        n_total_obs += n_observations[i];
    
    float* means = (float*) malloc(n_states*n_features*sizeof(float));
    float* variances = (float*) malloc(n_states*n_features*sizeof(float));
    float* sequences = (float*) malloc(n_total_obs*n_features*sizeof(float));
    float* loglikelihoods = (float*) malloc(n_total_obs*n_states*sizeof(float));
    float* _sequences = sequences;
    float* _loglikelihoods = loglikelihoods;
    for (i = 0; i < n_states*n_features; i++) {
        means[i] = i;
        variances[i] = (1+i);
    }

    for (s = 0; s < n_trajs; s++) {
        for (t = 0; t < n_observations[s]; t++)
            for (i = 0; i < n_features; i++)
                _sequences[t*n_features + i] = t - 10*i;
        _sequences += n_observations[s]*n_features;
    }

    gaussian_loglikelihood_diag(
            sequences,  means,  variances, n_trajs, &n_observations[0],
            n_states, n_features, loglikelihoods);

    // print sequences
    _sequences = sequences;
    _loglikelihoods = loglikelihoods;
    for (s = 0; s < n_trajs; s++) {
        printf("\n");
        for (t = 0; t < n_observations[s]; t++) {
            //for (i = 0; i < n_features; i++)
            //printf("%.2f ", _sequences[t*n_features + i]);
            //printf("\n");
            for (i = 0; i < n_states; i++)
                printf("%.3f ", _loglikelihoods[t*n_states + i]);
            printf("\n");
        }
        _sequences += n_observations[s]*n_features;
        _loglikelihoods += n_observations[s]*n_states;
    }
    // done printing

    /*
    import numpy as np
    from sklearn.mixture.gmm import _log_multivariate_normal_density_diag
    n_states = 4
    n_observations = np.array([2, 3, 4])
    n_features = 2
    n_total_obs = np.sum(n_observations)
    means = 1.0 * np.arange(n_states*n_features).reshape(n_states, n_features)
    variances = 1.0 + np.arange(n_states*n_features).reshape(n_states, n_features)
    for n in n_observations:
        s = np.arange(n)[:, np.newaxis] * np.ones((n, n_features)) - 10*np.arange(n_features)
        print _log_multivariate_normal_density_diag(s, means, variances)
        print
   */

    float expected[] = {
        -32.43445066, -24.87199706, -23.88847576, -24.48448148,
        -27.68445066, -21.24699706, -20.77180909, -21.6362672,
        -32.43445066, -24.87199706, -23.88847576, -24.48448148,
        -27.68445066, -21.24699706, -20.77180909, -21.6362672,
        -24.43445066, -18.20533039, -18.02180909, -19.05591005,
        -32.43445066, -24.87199706, -23.88847576, -24.48448148,
        -27.68445066, -21.24699706, -20.77180909, -21.6362672,
        -24.43445066, -18.20533039, -18.02180909, -19.05591005,
        -22.68445066, -15.74699706, -15.63847576, -16.74341005};

    for (i = 0; i < n_total_obs*n_states; i++)
        ASSERT_TOL(loglikelihoods[i], expected[i], 1e-4);
}
