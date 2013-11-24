/*****************************************************************/
/*    Copyright (c) 2013, Stanford University and the Authors    */
/*    Author: Robert McGibbon <rmcgibbo@gmail.com>               */
/*    Contributors:                                              */
/*                                                               */
/*****************************************************************/

#include <stdio.h>
#include "logsumexp.cu"
#ifndef FLT_MAX
#define FLT_MAX 1E+37
#endif

template<int BLOCK_SIZE>
__global__ void transitioncounts4(
const float* __restrict__ fwdlattice,
const float* __restrict__ bwdlattice,
const float* __restrict__ log_transmat,
const float* __restrict__ frame_logprob,
const int* __restrict__ sequence_lengths,
const int* __restrict__ cum_sequence_lengths,
const int n_trajs,
float* __restrict__  transcounts)
{
    const int N_STATES = 4;
    volatile __shared__ float logtmat[4][4];
    volatile __shared__ float logprob[BLOCK_SIZE/16];
    volatile __shared__ float fwd[BLOCK_SIZE/16][4][4];
    volatile __shared__ float bwd[BLOCK_SIZE/16][4][4];
    volatile __shared__ float flp[BLOCK_SIZE/16][4][4];

    unsigned int gid = blockIdx.x*blockDim.x+threadIdx.x;
    if (gid < 16)
        logtmat[gid/4][gid%4] = log_transmat[gid];


    while (gid/16 < n_trajs) {
        const unsigned int s = gid / 16;
        const unsigned int lid = gid % 16;
        const int i = lid / 4;
        const int j = lid % 4;
        const unsigned int teamid = (gid % BLOCK_SIZE) / 16;
        const float* _fwdlattice = fwdlattice + cum_sequence_lengths[s]*N_STATES;
        const float* _bwdlattice = bwdlattice + cum_sequence_lengths[s]*N_STATES;
        const float* _frame_logprob = frame_logprob + cum_sequence_lengths[s]*N_STATES;
        float lneta_ij = -FLT_MAX;

        if (lid < 4) {
            float tmp = _fwdlattice[(sequence_lengths[s]-1)*N_STATES + lid];
            tmp = logsumexp<4>(tmp);
            if (lid == 0)
                logprob[teamid] = tmp;
        }

        int tBlock;
        for (tBlock = 0; tBlock < ((sequence_lengths[s]-1)+2)/3; tBlock += 1) {
            /**
             * Load up blocks into shard memory of shape 4 x 4. Because the inner
             * loop requires knowning t and t+1, we only do t=0,1,2 during each
             * block iteration.
             **/
            if (lid == 0)
                printf("s=%d tBlock=%d\n", s, tBlock);
            if ((tBlock*3*N_STATES + lid) < sequence_lengths[s]*N_STATES) {
                fwd[teamid][i][j] = _fwdlattice[tBlock*3*N_STATES + lid];
                bwd[teamid][i][j] = _bwdlattice[tBlock*3*N_STATES + lid];
                flp[teamid][i][j] = _frame_logprob[tBlock*3*N_STATES + lid];
            } else {
                fwd[teamid][i][j] = -FLT_MAX;
                bwd[teamid][i][j] = -FLT_MAX;
                flp[teamid][i][j] = -FLT_MAX;
            }
            // not required since we're only warp-synchronous
            // __syncthreads();

            float Slneta_ij[3];
            #pragma unroll
            for (int t = 0; t < 3; t++)
                Slneta_ij[t] = fwd[teamid][t][i] + logtmat[i][j] + flp[teamid][t+1][j] + bwd[teamid][t+1][j] - logprob[teamid];

            float m = fmaxf(fmaxf(Slneta_ij[0], Slneta_ij[1]), fmaxf(Slneta_ij[2], lneta_ij));
            lneta_ij = m + logf(expf(Slneta_ij[0]-m) + expf(Slneta_ij[1]-m) +
                                expf(Slneta_ij[2]-m) + expf(lneta_ij-m));
        }
        atomicAdd(transcounts + i*4+j, expf(lneta_ij));
        gid += gridDim.x*blockDim.x;
    }
}
