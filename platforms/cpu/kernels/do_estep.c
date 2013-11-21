#include "stdlib.h"
#include "stdio.h"
#include "omp.h"
#include "math.h"

#include "gaussian_likelihood.h"
#include "forward.h"
#include "backward.h"
#include "posteriors.h"                 
#include "transitioncounts.h"
#include "sgemm.h"


void do_estep(const float* __restrict__ log_transmat,
              const float* __restrict__ log_transmat_T,
              const float* __restrict__ log_startprob,
              const float* __restrict__ means,
              const float* __restrict__ variances,
              const float** __restrict__ sequences,
              const int n_sequences,
              const int* __restrict__ sequence_lengths,
              const int n_features,
              const int n_states,
              float* __restrict__ transcounts,
              float* __restrict__ obs,
              float* __restrict__ obs2,
              float* __restrict__ post,
              float* logprob)
{
    int i, j, k;
    const float alpha = 1.0;
    const float beta = 1.0;
    const float *sequence;
    float *sequence2;
    float *means_over_variances, *means2_over_variances, *log_variances;
    float *framelogprob, *fwdlattice, *bwdlattice, *posteriors, *seq_transcounts, *seq_obs, *seq_obs2, *seq_post;
    
    means_over_variances = (float*) malloc(n_states*n_features*sizeof(float));
    means2_over_variances = (float*) malloc(n_states*n_features*sizeof(float));
    log_variances = (float*) malloc(n_states*n_features*sizeof(float));
    if (means2_over_variances == NULL || means2_over_variances == NULL || log_variances == NULL) {
        fprintf(stderr, "Memory allocation failure");
        exit(EXIT_FAILURE);
    }
    for (i = 0; i < n_states*n_features; i++) {
        means_over_variances[i] = means[i] / variances[i];
        means2_over_variances[i] = means_over_variances[i]*means[i];
        log_variances[i] = log(variances[i]);
    }

    #ifdef _OPENMP
    #pragma omp parallel for shared(means_over_variances, means2_over_variances, log_variances) \
        private(sequence, sequence2, framelogprob, fwdlattice, bwdlattice, posteriors, seq_transcounts, seq_obs, seq_obs2, seq_post)
    #endif
    for (i = 0; i < n_sequences; i++) {
        sequence = sequences[i];
        sequence2 = (float*) malloc(sequence_lengths[i]*n_features*sizeof(float));
        framelogprob = (float*) malloc(sequence_lengths[i]*n_states*sizeof(float));
        fwdlattice = (float*) malloc(sequence_lengths[i]*n_states*sizeof(float));
        bwdlattice = (float*) malloc(sequence_lengths[i]*n_states*sizeof(float));
        posteriors = (float*) malloc(sequence_lengths[i]*n_states*sizeof(float));
        seq_transcounts = (float*) calloc(n_states*n_states, sizeof(float));
        seq_obs = (float*) calloc(n_states*n_features, sizeof(float));
        seq_obs2 = (float*) calloc(n_states*n_features, sizeof(float));
        seq_post = (float*) calloc(n_states, sizeof(float));
        for (j = 0; j < sequence_lengths[i]*n_features; j++)
            sequence2[j] = sequence[j]*sequence[j];

        // Do work for this sequence
        gaussian_loglikelihood_diag(sequence, sequence2, means, variances,
                                    means_over_variances, means2_over_variances, log_variances,
                                    sequence_lengths[i], n_states, n_features, framelogprob);
        forward(log_transmat_T, log_startprob, framelogprob, sequence_lengths[i], n_states, fwdlattice);
        backward(log_transmat, log_startprob, framelogprob, sequence_lengths[i], n_states, bwdlattice);
        compute_posteriors(fwdlattice, bwdlattice, sequence_lengths[i], n_states, posteriors);

        // Compute sufficient statistics for this sequence
        transitioncounts(fwdlattice, bwdlattice, log_transmat, framelogprob, sequence_lengths[i], n_states, seq_transcounts, logprob);
        sgemm("N", "T", &n_features, &n_states, &sequence_lengths[i], &alpha, sequence, &n_features,
              posteriors, &n_states, &beta, seq_obs, &n_features);
        sgemm("N", "T", &n_features, &n_states, &sequence_lengths[i], &alpha, sequence2, &n_features,
              posteriors, &n_states, &beta, seq_obs2, &n_features);
        for (k = 0; k < n_states; k++)
            for (j = 0; j < sequence_lengths[i]; j++)
                seq_post[k] += posteriors[j*n_states + k];

        // Update the sufficient statistics. This needs to be threadsafe.
        for (j = 0; j < n_states; j++) {
            #ifdef _OPENMP
            #pragma omp atomic
            #endif
            post[j] += seq_post[j];
            for (k = 0; k < n_features; k++) {
                #ifdef _OPENMP
                #pragma omp atomic
                #endif
                obs[j*n_features+k] += seq_obs[j*n_features+k];
                #ifdef _OPENMP
                #pragma omp atomic
                #endif
                obs2[j*n_features+k] += seq_obs2[j*n_features+k];
            }
            for (k = 0; k < n_states; k++) {
                #ifdef _OPENMP
                #pragma omp atomic
                #endif
                transcounts[j*n_states+k] += seq_transcounts[j*n_states+k];
            }
        }

        // Free iteration-local memory
        free(sequence2);
        free(framelogprob);
        free(fwdlattice);
        free(bwdlattice);
        free(posteriors);
        free(seq_transcounts);
        free(seq_obs);
        free(seq_obs2);
        free(seq_post);
    }

    free(means_over_variances);
    free(means2_over_variances);
    free(log_variances);
}

