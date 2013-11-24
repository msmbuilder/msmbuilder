/*****************************************************************/
/*    Copyright (c) 2013, Stanford University and the Authors    */
/*    Author: Robert McGibbon <rmcgibbo@gmail.com>               */
/*    Contributors:                                              */
/*                                                               */
/*****************************************************************/

template <bool b> struct static_assert{};
template <> struct static_assert<true> { __device__ static void valid_expression() {}; };
__inline__ __device__ int divU(int numerator, int denominator){
    return (numerator+denominator-1)/denominator;
}


#include <stdio.h>
//#include "logsumexp.cu"
#ifndef FLT_MAX
#define FLT_MAX 1E+37
#endif



template <int N>
__device__ float logsumexp(float value)
{
    float max = value;
    for(int offset = 1; offset < N; offset <<= 1)
        max = fmaxf(max, __shfl_down(max, offset, N));
    for(int offset = 1; offset < N; offset <<= 1)
        max = __shfl_up(max, offset, N);

    value = expf(value - max);

    for(int offset = 1; offset < N; offset <<= 1)
        value += __shfl_down(value, offset, N);

    value = logf(value) + max;
    for(int offset = 1; offset < N; offset <<= 1)
        value = __shfl_up(value, offset, N);

    return value;
}

/**
 * Compute the number of expected transitions between states. This kernel is
 * for specifically N_STATES = 4
 *
 * A half-warp is used per trajectory in the dataset. This kernel should be
 * invoked with a 1-dimensional block whose size is supplied in the template
 * argument BLOCK_SIZE (necessary so that the shared memory can be allocated
 * correctly)
 */
template<const unsigned int N_STATES, const unsigned int BLOCK_SIZE>
__device__ void transitioncounts(
const float* __restrict__ fwdlattice,
const float* __restrict__ bwdlattice,
const float* __restrict__ log_transmat,
const float* __restrict__ frame_logprob,
const int* __restrict__ sequence_lengths,
const int* __restrict__ cum_sequence_lengths,
const int n_trajs,
float* __restrict__  transcounts)
{
    // N_STATES squared must be less than BLOCK_SIZE, because the logic
    // that uses the theads to collaborativly load data into shared memory
    // needs this
    static_assert<N_STATES*N_STATES <= BLOCK_SIZE>::valid_expression();
    // The number of states must be a power of 2
    static_assert<N_STATES && ((N_STATES & (N_STATES-1)) == 0)>::valid_expression();
    
    volatile __shared__ float logtmat[N_STATES][N_STATES];
    volatile __shared__ float logprob[BLOCK_SIZE/(N_STATES*N_STATES)];
    volatile __shared__ float fwd[BLOCK_SIZE/(N_STATES*N_STATES)][N_STATES][N_STATES];
    volatile __shared__ float bwd[BLOCK_SIZE/(N_STATES*N_STATES)][N_STATES][N_STATES];
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
        const float* _fwdlattice = fwdlattice + cum_sequence_lengths[s]*N_STATES;
        const float* _bwdlattice = bwdlattice + cum_sequence_lengths[s]*N_STATES;
        const float* _frame_logprob = frame_logprob + cum_sequence_lengths[s]*N_STATES;
        float lneta_ij = -FLT_MAX;

        if (lid < N_STATES) {
            float tmp = _fwdlattice[(sequence_lengths[s]-1)*N_STATES + lid];
            tmp = logsumexp<N_STATES>(tmp);
            if (lid == 0)
                logprob[teamid] = tmp;
        } 

        for (int tBlock = 0;
             tBlock < divU(sequence_lengths[s]-1, N_STATES-1)*(N_STATES-1);
             tBlock += (N_STATES-1))
        {
            if ((tBlock*N_STATES + lid) < sequence_lengths[s]*N_STATES) {
                fwd[teamid][i][j] = _fwdlattice[tBlock*N_STATES + lid];
                bwd[teamid][i][j] = _bwdlattice[tBlock*N_STATES + lid];
                flp[teamid][i][j] = _frame_logprob[tBlock*N_STATES + lid];
            } else {
                fwd[teamid][i][j] = -FLT_MAX;
                bwd[teamid][i][j] = -FLT_MAX;
                flp[teamid][i][j] = -FLT_MAX;
            }
            if (N_STATES > 4)
                // when N_STATES <= 4, we are implicltly warp-synchronous
                __syncthreads(); 

            float Slneta_ij[N_STATES-1];
            #pragma unroll
            for (int t = 0; t < N_STATES-1; t++)
                Slneta_ij[t] = fwd[teamid][t][i] + logtmat[i][j] + flp[teamid][t+1][j] + bwd[teamid][t+1][j] - logprob[teamid];

            float m = Slneta_ij[0];
            #pragma unroll
            for (int t = 1; t < N_STATES-1; t++)
                m = fmaxf(m, Slneta_ij[t]);
            
            float local_logsumexp = expf(lneta_ij - m);
            #pragma unroll
            for (int t = 0; t < N_STATES-1; t++)
                local_logsumexp += expf(Slneta_ij[t] - m);
            lneta_ij = m + logf(local_logsumexp);
        }
        atomicAdd(transcounts + (i*N_STATES + j), expf(lneta_ij));
        gid += gridDim.x*blockDim.x;
    }
}

extern "C" {
__global__ void transitioncounts4(
const float* __restrict__ fwdlattice,
const float* __restrict__ bwdlattice,
const float* __restrict__ log_transmat,
const float* __restrict__ frame_logprob,
const int* __restrict__ sequence_lengths,
const int* __restrict__ cum_sequence_lengths,
const int n_trajs,
float* __restrict__  transcounts) {
    transitioncounts<4, 64>(fwdlattice, bwdlattice, log_transmat, frame_logprob,
                     sequence_lengths, cum_sequence_lengths,  n_trajs,
                     transcounts);
}

__global__ void transitioncounts8(
const float* __restrict__ fwdlattice,
const float* __restrict__ bwdlattice,
const float* __restrict__ log_transmat,
const float* __restrict__ frame_logprob,
const int* __restrict__ sequence_lengths,
const int* __restrict__ cum_sequence_lengths,
const int n_trajs,
float* __restrict__  transcounts) {
    transitioncounts<8, 64>(fwdlattice, bwdlattice, log_transmat, frame_logprob,
                     sequence_lengths, cum_sequence_lengths,  n_trajs,
                     transcounts);
}

__global__ void transitioncounts16(
const float* __restrict__ fwdlattice,
const float* __restrict__ bwdlattice,
const float* __restrict__ log_transmat,
const float* __restrict__ frame_logprob,
const int* __restrict__ sequence_lengths,
const int* __restrict__ cum_sequence_lengths,
const int n_trajs,
float* __restrict__  transcounts) {
    transitioncounts<16, 256>(fwdlattice, bwdlattice, log_transmat, frame_logprob,
                     sequence_lengths, cum_sequence_lengths,  n_trajs,
                     transcounts);
}

}