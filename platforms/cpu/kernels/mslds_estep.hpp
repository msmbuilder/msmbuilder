/*****************************************************************/
/*    Copyright (c) 2013, Stanford University and the Authors    */
/*    Author: Robert McGibbon <rmcgibbo@gmail.com>               */
/*    Contributors: Bharath Ramsundar                            */
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
#include "cblas.h"

namespace Mixtape {

void _update_state_matrices(int k, int n, float* A, const float* alpha, const float* B)
{
    int kk, i;
    for (kk = 0; kk < k; kk++) {
        for (i = 0; i < n*n; i++)
            A[kk * n * n + i] += alpha[kk] * B[i];
    }
}

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
              const float* __restrict__ As,
              const float* __restrict__ bs,
              const float* __restrict__ Qs,
              const float* __restrict__ means,
              const float* __restrict__ covariances,
              const float** __restrict__ sequences,
              const int n_sequences,
              const int* __restrict__ sequence_lengths,
              const int n_features,
              const int n_states,
              const bool hmm_hotstart,
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
    int i, j, k, m, n, length, length_minus_1;
    float tlocallogprob;
    const float onef = 1.0;
    const float *sequence;
    float *framelogprob, *posteriors, *seq_transcounts, *seq_obs, *seq_obs_but_first;
    float *seq_obs_but_last, *seq_obs_obs_T, *seq_obs_obs_T_offset;
    float *seq_obs_obs_T_but_first, *seq_obs_obs_T_but_last, *seq_post;
    float *seq_post_but_last, *seq_post_but_first;
    float *frame_obs_obs_T;
    float obs_m, obs_n;

    REAL *fwdlattice, *bwdlattice;

    #ifdef _OPENMP
    #pragma omp parallel for default(none)                      \
        shared(log_transmat, log_transmat_T, log_startprob,     \
               As, bs, Qs, means, covariances, sequences,       \
               sequence_lengths, transcounts, obs,              \
               obs_but_first, obs_but_last,                     \
               obs_obs_T,obs_obs_T_offset,obs_obs_T_but_first,  \
               obs_obs_T_but_last, post, post_but_first,        \
               post_but_last, logprob, stderr)                  \
        private(sequence, framelogprob, fwdlattice, bwdlattice, \
                posteriors, seq_transcounts, seq_obs,           \
                seq_obs_but_first, seq_obs_but_last,            \
                seq_obs_obs_T, seq_obs_obs_T_offset,            \
                seq_obs_obs_T_but_first, seq_obs_obs_T_but_last,\
                frame_obs_obs_T, seq_post, seq_post_but_first,  \
                seq_post_but_last, tlocallogprob, j, k,         \
                length, length_minus_1, m, obs_m, n, obs_n)     
    #endif
    for (i = 0; i < n_sequences; i++) {
        printf("Analyzing sequence %d\n", i);
        sequence = sequences[i];
        length = sequence_lengths[i];
        length_minus_1 = length - 1;
        framelogprob = (float*) malloc(sequence_lengths[i]*n_states*sizeof(float));
        fwdlattice = (REAL*) malloc(sequence_lengths[i]*n_states*sizeof(REAL));
        bwdlattice = (REAL*) malloc(sequence_lengths[i]*n_states*sizeof(REAL));
        posteriors = (float*) malloc(sequence_lengths[i]*n_states*sizeof(float));
        seq_transcounts = (float*) calloc(n_states*n_states, sizeof(float));
        seq_obs = (float*) calloc(n_states*n_features, sizeof(float));
        seq_obs_but_first = (float*) calloc(n_states*n_features, sizeof(float));
        seq_obs_but_last = (float*) calloc(n_states*n_features, sizeof(float));
        seq_obs_obs_T = (float*) calloc(n_states*n_features*n_features, sizeof(float));
        seq_obs_obs_T_offset = (float*) calloc(n_states*n_features*n_features, sizeof(float));
        seq_obs_obs_T_but_first = (float*) calloc(n_states*n_features*n_features, sizeof(float));
        seq_obs_obs_T_but_last = (float*) calloc(n_states*n_features*n_features, sizeof(float));
        frame_obs_obs_T = (float*) malloc(n_features*n_features*sizeof(float));
        seq_post = (float*) calloc(n_states, sizeof(float));
        seq_post_but_first = (float*) calloc(n_states, sizeof(float));
        seq_post_but_last = (float*) calloc(n_states, sizeof(float));

        if (framelogprob == NULL || fwdlattice == NULL 
            || bwdlattice == NULL || posteriors == NULL
            || seq_transcounts == NULL || seq_obs == NULL 
            || seq_obs_obs_T ==NULL || seq_obs_obs_T_offset == NULL 
            || seq_obs_obs_T_but_first == NULL
            || seq_obs_obs_T_but_last == NULL || frame_obs_obs_T == NULL 
            || seq_post == NULL || seq_post_but_first == NULL 
            || seq_post_but_last == NULL) {
            fprintf(stderr, "Memory allocation failure in %s at %d\n", __FILE__, __LINE__); exit(EXIT_FAILURE);
        }

        // Compute the HMM log-likelihood if we're still warm-starting
        if (hmm_hotstart) {
            gaussian_loglikelihood_full(sequence, means, covariances,
                    length, n_states, n_features, framelogprob);
        } 
        // Else compute the MSLDS log likelihood 
        else {
            gaussian_lds_loglikelihood_full(sequence, As, bs, Qs, length,
                n_states, n_features, framelogprob);
        }
        forward(log_transmat_T, log_startprob, framelogprob, length,
                n_states, fwdlattice);
        backward(log_transmat, log_startprob, framelogprob, length,
                n_states, bwdlattice);
        compute_posteriors(fwdlattice, bwdlattice, length, n_states,
                posteriors); 

        // Compute sufficient statistics for this sequence
        tlocallogprob = 0;
        transitioncounts(fwdlattice, bwdlattice, log_transmat,
                framelogprob, sequence_lengths[i], n_states,
                seq_transcounts, &tlocallogprob);
        sgemm_("N", "T", &n_features, &n_states, &length, &onef, 
                sequence, &n_features, posteriors, &n_states, &onef,
                seq_obs, &n_features);
        sgemm_("N", "T", &n_features, &n_states, &length_minus_1, &onef,
                sequence, &n_features, posteriors, &n_states, &onef,
                seq_obs_but_last, &n_features);
        sgemm_("N", "T", &n_features, &n_states, &length_minus_1, &onef,
                sequence + n_features, &n_features, posteriors + n_states,
                &n_states, &onef, seq_obs_but_first, &n_features);

        for (k = 0; k < n_states; k++) {
            for (j = 0; j < sequence_lengths[i]; j++) {
                seq_post[k] += posteriors[j*n_states + k];
                if (j > 0)
                    seq_post_but_first[k] += posteriors[j*n_states + k];
                if (j < sequence_lengths[i] - 1)
                    seq_post_but_last[k] += posteriors[j*n_states + k];
            }
        }

        for (j = 0; j < sequence_lengths[i]; j++) {

            // sequence[j]*sequence[j].T
            for (m = 0; m < n_features; m++) {
                obs_m = sequence[j*n_features + m];
                for (int n = 0; n < n_features; n++) {
                    obs_n = sequence[j*n_features + n];
                    frame_obs_obs_T[m*n_features + n] = obs_m*obs_n;
                }
            }

            _update_state_matrices(n_states, n_features, seq_obs_obs_T,
                    &posteriors[j*n_states], frame_obs_obs_T);
            if (j > 0)
                _update_state_matrices(n_states, n_features,
                        seq_obs_obs_T_but_first, &posteriors[j*n_states],
                        frame_obs_obs_T);
            if (j < sequence_lengths[i] - 1)
                _update_state_matrices(n_states, n_features,
                        seq_obs_obs_T_but_last, &posteriors[j*n_states],
                        frame_obs_obs_T);

            if (j > 0) {
                // sequence[j]*sequence[j-1].T
                for (m = 0; m < n_features; m++) {
                    obs_m = sequence[j*n_features + m];
                    for (n = 0; n < n_features; n++) {
                        obs_n = sequence[(j-1)*n_features + n];
                        frame_obs_obs_T[m*n_features + n] = obs_m*obs_n;
                    }
                }
                _update_state_matrices(n_states, n_features,
                        seq_obs_obs_T_offset, &posteriors[j*n_states],
                        frame_obs_obs_T);
            }
        }

        // Update the sufficient statistics. This needs to be threadsafe.
        #ifdef _OPENMP
        #pragma omp critical
        {
        #endif
        *logprob += tlocallogprob;
        for (j = 0; j < n_states; j++) {
            post[j] += seq_post[j];
            post_but_first[j] += seq_post_but_first[j];
            post_but_last[j] += seq_post_but_last[j];

            for (k = 0; k < n_features; k++) {
                obs[j*n_features+k] += seq_obs[j*n_features+k];
                obs_but_first[j*n_features+k] += seq_obs_but_first[j*n_features+k];
                obs_but_last[j*n_features+k] += seq_obs_but_last[j*n_features+k];
            }

            for (k = 0; k < n_states; k++)
                transcounts[j*n_states+k] += seq_transcounts[j*n_states+k];

            for (k = 0; k < n_features*n_features; k++) {
                obs_obs_T[j*n_features*n_features + k] += seq_obs_obs_T[j*n_features*n_features + k];
                obs_obs_T_offset[j*n_features*n_features + k] += seq_obs_obs_T_offset[j*n_features*n_features + k];
                obs_obs_T_but_first[j*n_features*n_features + k] += seq_obs_obs_T_but_first[j*n_features*n_features + k];
                obs_obs_T_but_last[j*n_features*n_features + k] += seq_obs_obs_T_but_last[j*n_features*n_features + k];
            }
        }
        #ifdef _OPENMP
        }
        #endif

        // Free iteration-local memory
        free(framelogprob);
        free(fwdlattice);
        free(bwdlattice);
        free(posteriors);
        free(seq_transcounts);
        free(seq_obs);
        free(seq_obs_but_first);
        free(seq_obs_but_last);
        free(seq_obs_obs_T);
        free(seq_obs_obs_T_offset);
        free(seq_obs_obs_T_but_first);
        free(seq_obs_obs_T_but_last);
        free(seq_post);
        free(seq_post_but_first);
        free(seq_post_but_last);
        free(frame_obs_obs_T);
    }

}

} // namespace

#endif
