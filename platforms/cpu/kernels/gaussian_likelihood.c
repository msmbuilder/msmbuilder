/*****************************************************************/
/*    Copyright (c) 2013, Stanford University and the Authors    */
/*    Author: Robert McGibbon <rmcgibbo@gmail.com>               */
/*    Contributors:                                              */
/*                                                               */
/*****************************************************************/

#include "math.h"
#include "stdio.h"
#include "stdlib.h"
#include "string.h"
#include "gaussian_likelihood.h"

extern int spotrf_(const char *uplo, const int *n, float *a, const int *lda, int *info);
extern int strtrs_(const char *uplo, const char *trans, const char *diag, const int *n, 
    const int *nrhs, const float *a, const int *lda, float *b, const int *ldb, int * info);

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


void gaussian_loglikelihood_full(const float* __restrict__ sequence,
                                 const float* __restrict__ means,
                                 const float* __restrict__ covariances,
                                 const int n_observations,
                                 const int n_states,
                                 const int n_features,
                                 float* __restrict__ loglikelihoods)
{
    int i, j, k, info;
    float chol_sol, chol2, cv_log_det;
    float* cv_chol;
    float* sequence_minus_means = malloc(n_observations * n_features * sizeof(float));
    float prefactor = n_features * log(2 * M_PI);

    for (i = 0; i < n_states; i++) {
        cv_chol = malloc(n_features * n_features * sizeof(float));
        memcpy(cv_chol, &covariances[i*n_features*n_features], n_features*n_features*sizeof(float));
        for (j = 0; j < n_observations; j++)
            for (k = 0; k < n_features; k++)
                sequence_minus_means[j*n_features + k] = sequence[j*n_features + k] - means[i*n_states + k];

        // Cholesky decomposition of the covariance matrix
        spotrf_("L", &n_features, cv_chol, &n_features, &info);
        if (info != 0) { fprintf(stderr, "LAPACK Error"); exit(1); }

        cv_log_det = 0;
        for (j = 0; j < n_features; j++) {
            cv_log_det += 2*log(cv_chol[j*n_features + j]);
        }

        // solve the triangular system
        strtrs_("L", "N", "N", &n_features, &n_observations, cv_chol, &n_features,
                sequence_minus_means, &n_features, &info);
        if (info != 0) { fprintf(stderr, "LAPACK Error"); exit(1); }

        for (j = 0; j < n_observations; j++) {
            loglikelihoods[i*n_states + j] = -0.5 * (cv_log_det + prefactor);
            for (k = 0; k < n_features; k++) {
                chol_sol = sequence_minus_means[j*n_features + k];
                chol2 = chol_sol * chol_sol;
                loglikelihoods[i*n_states + j] += -0.5*chol2;
            }
        }
        free(cv_chol);
    }
    free(sequence_minus_means);

}
    