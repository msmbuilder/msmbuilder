/*****************************************************************/
/*    Copyright (c) 2013, Stanford University and the Authors    */
/*    Author: Robert McGibbon <rmcgibbo@gmail.com>               */
/*    Contributors:                                              */
/*                                                               */
/*****************************************************************/

#include <stdio.h>
#include "logsumexp.cu"
#include "staticassert.cuh"

#ifndef FLT_MAX
#define FLT_MAX 1E+37
#endif


/**
 * Compute the number of expected transitions between states. This kernel is
 * for specifically N_STATES = 4, 8, 16. To run it with 32 states, you would
 * need a thread block with 32**2 state, which is not available.
 */
template <unsigned int N_STATES, unsigned int BLOCK_SIZE>
__global__ void transitioncounts4_8_16(
const mixed* __restrict__ fwdlattice,
const mixed* __restrict__ bwdlattice,
const float* __restrict__ log_transmat,
const float* __restrict__ frame_logprob,
const int* __restrict__ sequence_lengths,
const int* __restrict__ cum_sequence_lengths,
const int n_trajs,
float* __restrict__  transcounts,
float* __restrict__ logprob)
{
    // N_STATES squared must be less than BLOCK_SIZE, because the logic
    // that uses the theads to collaborativly load data into shared memory
    // needs this
    static_assert<N_STATES*N_STATES <= BLOCK_SIZE>::valid_expression();
    // The number of states must be a power of 2.
    static_assert<N_STATES && ((N_STATES & (N_STATES-1)) == 0)>::valid_expression();

    volatile __shared__ float logtmat[N_STATES][N_STATES];
    volatile __shared__ mixed tlogprob[BLOCK_SIZE/(N_STATES*N_STATES)];
    volatile __shared__ mixed fwd[BLOCK_SIZE/(N_STATES*N_STATES)][N_STATES][N_STATES];
    volatile __shared__ mixed bwd[BLOCK_SIZE/(N_STATES*N_STATES)][N_STATES][N_STATES];
    volatile __shared__ float flp[BLOCK_SIZE/(N_STATES*N_STATES)][N_STATES][N_STATES];

    unsigned int gid = blockIdx.x*blockDim.x+threadIdx.x;
    if (gid < N_STATES*N_STATES)
        logtmat[gid/N_STATES][gid%N_STATES] = log_transmat[gid];

    while (gid/(N_STATES*N_STATES) < n_trajs) {
        const unsigned int s = gid / (N_STATES*N_STATES);
        const unsigned int lid = gid % (N_STATES*N_STATES);
        const int i = lid / N_STATES;
        const int j = lid % N_STATES;
        const unsigned int teamid = (gid % BLOCK_SIZE) / (N_STATES*N_STATES);
        const mixed* _fwdlattice = fwdlattice + cum_sequence_lengths[s]*N_STATES;
        const mixed* _bwdlattice = bwdlattice + cum_sequence_lengths[s]*N_STATES;
        const float* _frame_logprob = frame_logprob + cum_sequence_lengths[s]*N_STATES;
        mixed lneta_ij = -FLT_MAX;

        if (lid < N_STATES) {
            mixed tmp = _fwdlattice[(sequence_lengths[s]-1)*N_STATES + lid];
            tmp = logsumexp<mixed, N_STATES>(tmp);
            if (lid == 0) {
                tlogprob[teamid] = tmp;
                atomicAdd(logprob, (float) tlogprob[teamid]);
            }
        }

        for (int tBlock = 0;
             tBlock < divU(sequence_lengths[s]-1, N_STATES-1)*(N_STATES-1);
             tBlock += (N_STATES-1))
        {
            if (tBlock > 7) break;
            
            if (N_STATES > 4)
                // After 2 hours of very painful debugging, it seems that if 
                // this sync isnt here *before* the load into shared memory,
                // some of the threads in this iteration can kill other threads
                // in the previous iteration
                __syncthreads();

            if ((tBlock*N_STATES + lid) < sequence_lengths[s]*N_STATES) {
                fwd[teamid][i][j] = _fwdlattice[tBlock*N_STATES + lid];
                bwd[teamid][i][j] = _bwdlattice[tBlock*N_STATES + lid];
                flp[teamid][i][j] = _frame_logprob[tBlock*N_STATES + i*N_STATES + j];
            } else {
                fwd[teamid][i][j] = -FLT_MAX;
                bwd[teamid][i][j] = -FLT_MAX;
                flp[teamid][i][j] = -FLT_MAX;
            }

            if (N_STATES > 4)
                // when N_STATES <= 4, we are implicltly warp-synchronous
                __syncthreads();

            mixed Slneta_ij[N_STATES-1];
            #pragma unroll
            for (int t = 0; t < N_STATES-1; t++) {
                Slneta_ij[t] = fwd[teamid][t][i] + logtmat[i][j] + flp[teamid][t+1][j] + bwd[teamid][t+1][j] - tlogprob[teamid];
            }

            mixed m = lneta_ij;
            #pragma unroll
            for (int t = 0; t < N_STATES-1; t++)
                m = fmaxf(m, Slneta_ij[t]);

            mixed local_logsumexp = expf(lneta_ij - m);
            #pragma unroll
            for (int t = 0; t < N_STATES-1; t++)
                local_logsumexp += expf(Slneta_ij[t] - m);
            lneta_ij = m + log(local_logsumexp);
        }
        atomicAdd(transcounts + (i*N_STATES + j), (float) exp(lneta_ij));
        gid += gridDim.x*blockDim.x;
    }
}

