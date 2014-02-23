/*****************************************************************/
/*    Copyright (c) 2013, Stanford University and the Authors    */
/*    Author: Robert McGibbon <rmcgibbo@gmail.com>               */
/*    Contributors:                                              */
/*                                                               */
/*****************************************************************/
#ifndef MIXTAPE_CPU_MSLDS_ESTEP
#define MIXTAPE_CPU_MSLDS_ESTEP

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
#include "sgemm.h"

// sger_(&n_features, &n_features, &onef, &sequence[j*n_features], &one,
//       sequence[j*n_features], &one, seq_obs_obs_T[j*n_features*n_features],
//       &n_features);
// extern int sger_(int *m, int *n, float *alpha,  float *x, int *incx, float *y,
//                  int *incy, float *a, int *lda);

namespace Mixtape {

/**
 * Run the Metastable Switching Linear Dynamical System E-step, computing
 * sufficient statistics over all of the trajectories
 *
 * The template parameter controls the precision of the foward and backward
 * lattices which are subject to accumulated floating point error during long
 * trajectories.
 */
template<typename REAL>
void do_mslds_estep(const float* __restrict__ log_transmat,
              const float* __restrict__ log_transmat_T,
              const float* __restrict__ log_startprob,
              const float* __restrict__ means,
              const float* __restrict__ covariances,
              const float** __restrict__ sequences,
              const int n_sequences,
              const int* __restrict__ sequence_lengths,
              const int n_features,
              const int n_states,
              float* __restrict__ transcounts,
              float* __restrict__ obs,
              float* __restrict__ obs_but_first,
              float* __restrict__ obs_but_last,
              float* __restrict__ obs_obs_T,
              float* __restrict__ obs_obs_T_offset,
              float* __restrict__ obs_obs_T_but_first,
              float* __restrict__ obs_obs_T_but_last,
              float* __restrict__ post,
              float* __restrict__ post_but_first,
              float* __restrict__ post_but_last,
              float* __restrict__ logprob)
{
    int i, j, k, m, n;
    float tlocallogprob;
    const int one = 1;
    const float onef = 1.0;
    const float *sequence;
    float *framelogprob, *posteriors, *seq_transcounts, *seq_obs, *seq_obs_but_first;
    float *seq_obs_but_last, *seq_obs_obs_T, *seq_obs_obs_T_offset;
    float *seq_obs_obs_T_but_first, *seq_obs_obs_T_but_last, *seq_post;
    float *seq_post_but_last, *seq_post_but_first;
    float obs_m, obs_n;
    
    REAL *fwdlattice, *bwdlattice;
    // 

    // #ifdef _OPENMP
    // #pragma omp parallel for default(none)                                  \
    //     shared(log_transmat, log_transmat_T, log_startprob, means,          \
    //            variances, sequences, sequence_lengths, transcounts,         \
    //            obs, obs2, post, logprob, means_over_variances,              \
    //            means2_over_variances, log_variances, stderr)                \
    //     private(sequence, sequence2, framelogprob, fwdlattice, bwdlattice,  \
    //             posteriors, seq_transcounts, seq_obs, seq_obs2, seq_post,   \
    //             tlocallogprob, j, k)
    // #endif
    for (i = 0; i < n_sequences; i++) {
        sequence = sequences[i];
        framelogprob = (float*) malloc(sequence_lengths[i]*n_states*sizeof(float));
        fwdlattice = (REAL*) malloc(sequence_lengths[i]*n_states*sizeof(REAL));
        bwdlattice = (REAL*) malloc(sequence_lengths[i]*n_states*sizeof(REAL));
        posteriors = (float*) malloc(sequence_lengths[i]*n_states*sizeof(float));
        seq_transcounts = (float*) calloc(n_states*n_states, sizeof(float));
        seq_obs = (float*) calloc(n_states*n_features, sizeof(float));
        seq_obs_obs_T = (float*) calloc(n_states*n_features*n_features, sizeof(float));
        // seq_obs = (float*) calloc(n_states*n_features, sizeof(float));
        // seq_obs = (float*) calloc(n_states*n_features, sizeof(float));
        
        // seq_obs2 = (float*) calloc(n_states*n_features, sizeof(float));
        seq_post = (float*) calloc(n_states, sizeof(float));
        // if (sequence2==NULL || framelogprob == NULL || fwdlattice == NULL || bwdlattice == NULL || posteriors == NULL
            // || seq_transcounts == NULL || seq_obs == NULL || seq_obs2 ==NULL || seq_post == NULL) {
            // fprintf(stderr, "Memory allocation failure in %s at %d\n", __FILE__, __LINE__); exit(EXIT_FAILURE);
        // }

        // Do work for this sequence
        gaussian_loglikelihood_full(sequence, means, covariances,
            sequence_lengths[i], n_states, n_features,framelogprob);
        forward(log_transmat_T, log_startprob, framelogprob, sequence_lengths[i], n_states, fwdlattice);
        backward(log_transmat, log_startprob, framelogprob, sequence_lengths[i], n_states, bwdlattice);
        compute_posteriors(fwdlattice, bwdlattice, sequence_lengths[i], n_states, posteriors);
    
        // Compute sufficient statistics for this sequence
        tlocallogprob = 0;
        transitioncounts(fwdlattice, bwdlattice, log_transmat, framelogprob, sequence_lengths[i], n_states, seq_transcounts, &tlocallogprob);
        sgemm("N", "T", &n_features, &n_states, &sequence_lengths[i], &onef, sequence, &n_features, posteriors, &n_states, &onef, seq_obs, &n_features);
        for (k = 0; k < n_states; k++)
            for (j = 0; j < sequence_lengths[i]; j++)
                seq_post[k] += posteriors[j*n_states + k];

        for (j = 0; j < sequence_lengths[i]; j++) {
            for (m = 0; m < n_features; m++) {
                obs_m = sequence[j*n_features + m];
                for (n = 0; n < n_features; n++) {
                    obs_n = sequence[j*n_features + n];
                    
                        for (k = 0; k < n_states; k++)
                            seq_obs_obs_T[k*n_features*n_features + m*n_features + n] += \
                                posteriors[j*n_states + k] * obs_m * obs_n;
                }
            }
        }
            
    //     // Update the sufficient statistics. This needs to be threadsafe.
    //     #ifdef _OPENMP
    //     #pragma omp critical
    //     {
    //     #endif
    //     *logprob += tlocallogprob;
    //     for (j = 0; j < n_states; j++) {
    //         post[j] += seq_post[j];
    //         for (k = 0; k < n_features; k++) {
    //             obs[j*n_features+k] += seq_obs[j*n_features+k];
    //             obs2[j*n_features+k] += seq_obs2[j*n_features+k];
    //         }
    //         for (k = 0; k < n_states; k++) {
    //             transcounts[j*n_states+k] += seq_transcounts[j*n_states+k];
    //         }
    //     }
    //     #ifdef _OPENMP
    //     }
    //     #endif
    // 
    //     // Free iteration-local memory
        free(framelogprob);
        free(fwdlattice);
        free(bwdlattice);
        free(posteriors);
        free(seq_transcounts);
        free(seq_obs);
        free(seq_obs_obs_T);
        free(seq_post);
    }
    // 
    // free(means_over_variances);
    // free(means2_over_variances);
    // free(log_variances);
}


} // namespace

#endif
