/*******************************************************************/
/*    Copyright (c) 2013, Stanford University and the Authors      */
/*    Author: Robert McGibbon <rmcgibbo@gmail.com>                 */
/*    Contributors: Bharath Ramsundar <bharath.ramsundar@gmail.com */
/*                                                                 */
/*******************************************************************/

#include "math.h"
#include "stdio.h"
#include "stdlib.h"
#include "string.h"
#include "gaussian_likelihood.h"
#include "cblas.h"

void gaussian_loglikelihood_diag(const float* __restrict__ sequence,
                                 const double* __restrict__ means,
                                 const double* __restrict__ variances,
                                 const double* __restrict__ means_over_variances,
                                 const double* __restrict__ means2_over_variances,
                                 const double* __restrict__ log_variances,
                                 const int n_observations,
                                 const int n_states, const int n_features,
                                 float* __restrict__ loglikelihoods)
{
    int t, i, j;
    double temp;
    double x;
    static const double log_M_2_PI = 1.8378770664093453; // np.log(2*np.pi)

    for (t = 0; t < n_observations; t++) {
        for (j = 0; j < n_states; j++) {
            temp = 0.0f;
            for (i = 0; i < n_features; i++) {
                x = sequence[t*n_features + i];
                temp += means2_over_variances[j*n_features + i]              \
                        - 2.0 * x * means_over_variances[j*n_features + i]   \
                        + (x * x) / variances[j*n_features + i]              \
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
    if (sequence_minus_means == NULL) {
        fprintf(stderr, "Memory allocation failure in %s at %d\n",
                __FILE__, __LINE__); 
        exit(EXIT_FAILURE);
    }

    for (i = 0; i < n_states; i++) {
        cv_chol = malloc(n_features * n_features * sizeof(float));
        if (cv_chol == NULL) {
            fprintf(stderr, "Memory allocation failure in %s at %d\n",
                    __FILE__, __LINE__); 
            exit(EXIT_FAILURE);
        }
        memcpy(cv_chol, &covariances[i*n_features*n_features], n_features*n_features*sizeof(float));
        for (j = 0; j < n_observations; j++)
            for (k = 0; k < n_features; k++)
                sequence_minus_means[j*n_features + k] = sequence[j*n_features + k] - means[i*n_features + k];

        // Cholesky decomposition of the covariance matrix
        spotrf_("L", &n_features, cv_chol, &n_features, &info);
        if (info != 0) { fprintf(stderr, "LAPACK Error in %s at %d\n", __FILE__, __LINE__); exit(1); }

        cv_log_det = 0;
        for (j = 0; j < n_features; j++) {
            cv_log_det += 2*log(cv_chol[j*n_features + j]);
        }

        // solve the triangular system
        strtrs_("L", "N", "N", &n_features, &n_observations, cv_chol, &n_features,
                sequence_minus_means, &n_features, &info);
        if (info != 0) { 
            fprintf(stderr, "LAPACK Error in %s at %d\n", __FILE__,
                    __LINE__); 
            exit(1); 
        }

        for (j = 0; j < n_observations; j++) {
            loglikelihoods[j*n_states + i] = -0.5 * (cv_log_det + prefactor);
            for (k = 0; k < n_features; k++) {
                chol_sol = sequence_minus_means[j*n_features + k];
                chol2 = chol_sol * chol_sol;
                loglikelihoods[j*n_states + i] += -0.5*chol2;
            }
        }
        free(cv_chol);
    }
    free(sequence_minus_means);
}

void gaussian_lds_loglikelihood_full(const float* __restrict__ sequence,
                                     const float* __restrict__ As,
                                     const float* __restrict__ bs,
                                     const float* __restrict__ Qs,
                                     const int n_observations,
                                     const int n_states,
                                     const int n_features,
                                     float* __restrict__ lds_loglikelihoods)
{
    int i, j, k, info;
    float chol_sol, chol2;
    float* sequence_minus_pred = malloc(n_observations * n_features * sizeof(float));
    float prefactor = n_features * log(2 * M_PI);
    float alpha = 1.0;
    float beta = 1.0;
    float Q_i_log_det;
    float* Q_i;
    float* A_i;
    float* b_i;

    for (i = 0; i < n_states; i++) {
        Q_i = malloc(n_features * n_features * sizeof(float));
        A_i = malloc(n_features * n_features * sizeof(float));
        b_i = malloc(n_features * n_observations * sizeof(float));
        if (Q_i == NULL || A_i == NULL || b_i == NULL) {
            fprintf(stderr, "Memory allocation failure in %s at %d\n",
                    __FILE__, __LINE__); 
            exit(EXIT_FAILURE);
        }
        memcpy(Q_i, &Qs[i*n_features*n_features], n_features*n_features*sizeof(float));
        memcpy(A_i, &As[i*n_features*n_features], n_features*n_features*sizeof(float));
        for (j =  0; j < n_observations; j++) {
            memcpy(&b_i[j*n_features], &bs[i*n_features], n_features*sizeof(float));
        }
        // Compute b_i := A_i * sequence[j] + b_i for all j
        sgemm_("N", "N", &n_features, &n_observations, &n_features, &alpha, A_i, &n_features, sequence, &n_features, &beta, b_i, &n_features);
        for (j = 0; j < n_observations; j++) {
            for (k = 0; k < n_features; k++) {
                sequence_minus_pred[j*n_features+k] = sequence[j*n_features+k] - b_i[j*n_features + k];
            }
        }

        // Cholesky decomposition of covariance matrix
        spotrf_("L", &n_features, Q_i, &n_features, &info);
        if (info != 0) { 
            fprintf(stderr, "LAPACK Error in %s at %d in iteration %d\n", __FILE__, __LINE__, i); 
            exit(1); 
        }

        // Compute the log-det of the covariance matrix down the diagonal
        Q_i_log_det = 0;
        for (j = 0; j < n_features; j++) {
            Q_i_log_det += 2*log(Q_i[j*n_features + j]);
        }

        // Solve the triangular system
        strtrs_("L", "N", "N", &n_features, &n_observations, Q_i, &n_features, sequence_minus_pred, &n_features, &info);
        if (info != 0) { 
            fprintf(stderr, "LAPACK Error in %s at %d\n", __FILE__, __LINE__); 
            exit(1); 
        }

        for (j = 0; j < n_observations; j++) {
            lds_loglikelihoods[j*n_states + i] = -0.5 * (Q_i_log_det + prefactor);
            for (k = 0; k < n_features; k++) {
                chol_sol = sequence_minus_pred[j*n_features + k];
                chol2 = chol_sol * chol_sol;
                lds_loglikelihoods[j*n_states + i] += -0.5*chol2;
            }
        }
        free(Q_i);
        free(A_i);
        free(b_i);
    }
    free(sequence_minus_pred);
}
