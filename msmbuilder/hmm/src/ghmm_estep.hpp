/*****************************************************************/
/*    Copyright (c) 2013, Stanford University and the Authors    */
/*    Author: Robert McGibbon <rmcgibbo@gmail.com>               */
/*    Contributors:                                              */
/*                                                               */
/*****************************************************************/
#ifndef MIXTAPE_CPU_GHMM_ESTEP
#define MIXTAPE_CPU_GHMM_ESTEP

#include "stdlib.h"
#include "stdio.h"
#ifdef _OPENMP
#include "omp.h"
#endif
#include "math.h"

#include "gaussian_likelihood.h"
#include "forward.hpp"
#include "backward.hpp"
#include "posteriors.hpp"
#include "transitioncounts.hpp"
#include "scipy_lapack.h"

namespace Mixtape {

/**
 * Run the GHMM E-step, computing sufficient statistics over all of the trajectories
 *
 * The template parameter controls the precision of the foward and backward lattices
 * which are subject to accumulated floating point error during long trajectories.
 */
template<typename REAL>
void do_ghmm_estep(const float* __restrict log_transmat,
              const float* __restrict log_transmat_T,
              const float* __restrict log_startprob,
              const float* __restrict means,
              const float* __restrict variances,
              const float** __restrict sequences,
              const int n_sequences,
              const int* __restrict sequence_lengths,
              const int n_features,
              const int n_states,
              float* __restrict transcounts,
              float* __restrict obs,
              float* __restrict obs2,
              float* __restrict post,
              float* logprob)
{
    int i, j, k;
    float tlocallogprob;
    const float alpha = 1.0;
    const float beta = 1.0;
    const float *sequence;
    float *sequence2;
    float *means_over_variances, *means2_over_variances, *log_variances;
    float *framelogprob, *posteriors, *seq_transcounts, *seq_obs, *seq_obs2, *seq_post;
    REAL *fwdlattice, *bwdlattice;
    lapack_t *lapack = get_lapack();
    sgemm_t *sgemm = lapack->sgemm;

    means_over_variances = (float*) malloc(n_states*n_features*sizeof(float));
    means2_over_variances = (float*) malloc(n_states*n_features*sizeof(float));
    log_variances = (float*) malloc(n_states*n_features*sizeof(float));
    if (means2_over_variances == NULL || means2_over_variances == NULL || log_variances == NULL) {
        fprintf(stderr, "Memory allocation failure in %s at %d\n", __FILE__, __LINE__); exit(EXIT_FAILURE);
    }
    for (i = 0; i < n_states*n_features; i++) {
        means_over_variances[i] = means[i] / variances[i];
        means2_over_variances[i] = means_over_variances[i]*means[i];
        log_variances[i] = log(variances[i]);
    }

    #ifdef _OPENMP
    #pragma omp parallel for default(none) \
        shared(log_transmat, log_transmat_T, log_startprob, means, \
               variances, sequences, sequence_lengths, transcounts, \
               obs, obs2, post, logprob, means_over_variances, \
               means2_over_variances, log_variances, stderr, sgemm) \
        private(sequence, sequence2, framelogprob, fwdlattice, \
                bwdlattice, posteriors, seq_transcounts, seq_obs, \
                seq_obs2, seq_post, tlocallogprob, j, k)
    #endif
    for (i = 0; i < n_sequences; i++) {
        sequence = sequences[i];
        sequence2 = (float*) malloc(sequence_lengths[i]*n_features*sizeof(float));
        framelogprob = (float*) malloc(sequence_lengths[i]*n_states*sizeof(float));
        fwdlattice = (REAL*) malloc(sequence_lengths[i]*n_states*sizeof(REAL));
        bwdlattice = (REAL*) malloc(sequence_lengths[i]*n_states*sizeof(REAL));
        posteriors = (float*) malloc(sequence_lengths[i]*n_states*sizeof(float));
        seq_transcounts = (float*) calloc(n_states*n_states, sizeof(float));
        seq_obs = (float*) calloc(n_states*n_features, sizeof(float));
        seq_obs2 = (float*) calloc(n_states*n_features, sizeof(float));
        seq_post = (float*) calloc(n_states, sizeof(float));
        if (sequence2==NULL || framelogprob == NULL || fwdlattice == NULL || bwdlattice == NULL || posteriors == NULL
            || seq_transcounts == NULL || seq_obs == NULL || seq_obs2 ==NULL || seq_post == NULL) {
            fprintf(stderr, "Memory allocation failure in %s at %d\n", __FILE__, __LINE__); exit(EXIT_FAILURE);
        }

        for (j = 0; j < sequence_lengths[i]*n_features; j++)
            sequence2[j] = sequence[j]*sequence[j];

        // Do work for this sequence
        gaussian_loglikelihood_diag(sequence, sequence2, means, variances,
                                    means_over_variances, means2_over_variances, log_variances,
                                    sequence_lengths[i], n_states, n_features,framelogprob);

        forward(log_transmat_T, log_startprob, framelogprob, sequence_lengths[i], n_states, fwdlattice);
        backward(log_transmat, log_startprob, framelogprob, sequence_lengths[i], n_states, bwdlattice);
        compute_posteriors(fwdlattice, bwdlattice, sequence_lengths[i], n_states, posteriors);

        // Compute sufficient statistics for this sequence
        tlocallogprob = 0;
        transitioncounts(fwdlattice, bwdlattice, log_transmat, framelogprob, sequence_lengths[i], n_states, seq_transcounts, &tlocallogprob);
        sgemm("N", "T", &n_features, &n_states, &sequence_lengths[i], &alpha, sequence, &n_features, posteriors, &n_states, &beta, seq_obs, &n_features);
        sgemm("N", "T", &n_features, &n_states, &sequence_lengths[i], &alpha, sequence2, &n_features, posteriors, &n_states, &beta, seq_obs2, &n_features);
        for (k = 0; k < n_states; k++)
            for (j = 0; j < sequence_lengths[i]; j++)
                seq_post[k] += posteriors[j*n_states + k];

        // Update the sufficient statistics. This needs to be threadsafe.
        #ifdef _OPENMP
        #pragma omp critical
        {
        #endif
        *logprob += tlocallogprob;
        for (j = 0; j < n_states; j++) {
            post[j] += seq_post[j];
            for (k = 0; k < n_features; k++) {
                obs[j*n_features+k] += seq_obs[j*n_features+k];
                obs2[j*n_features+k] += seq_obs2[j*n_features+k];
            }
            for (k = 0; k < n_states; k++) {
                transcounts[j*n_states+k] += seq_transcounts[j*n_states+k];
            }
        }
        #ifdef _OPENMP
        }
        #endif

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


} // namespace

#endif
