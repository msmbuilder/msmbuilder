/*****************************************************************/
/*    Copyright (c) 2013, Stanford University and the Authors    */
/*    Author: Robert McGibbon <rmcgibbo@gmail.com>               */
/*    Contributors:                                              */
/*                                                               */
/*****************************************************************/

#include "logsumexp.cu"
#include "stdio.h"
#include <stdlib.h>

__global__ void square(const float* __restrict__ in, int n, float* __restrict__ out) {
    int gid = blockIdx.x*blockDim.x + threadIdx.x;
    while (gid < n) {
        out[gid] = in[gid] * in[gid];
        gid += blockDim.x*gridDim.x;
    }
}

__global__ void fill(float* __restrict__ devPtr, float value, int n) {
    int gid = blockIdx.x*blockDim.x + threadIdx.x;
    while (gid < n) {
        devPtr[gid] = value;
        gid += blockDim.x*gridDim.x;
    }
}


__global__ void gaussian_likelihood(
const float* __restrict__ sequences,
const float* __restrict__ means,
const float* __restrict__ variances,
const int n_sequences,
const int* __restrict__ sequence_lengths,
const int* __restrict__ cum_sequence_lengths,
const int n_states,
const int n_features,
float* __restrict__ loglikelihoods)
{
    const unsigned int WARPS_PER_TRAJ = 4;
    const unsigned int WARP_WIDTH = 32;
    const unsigned int FEATURE_WIDTH = ((n_features + WARP_WIDTH - 1) / WARP_WIDTH) * WARP_WIDTH;
    const float log_M_2_PI = 1.8378770664093453f;
    unsigned int gid = blockIdx.x*blockDim.x+threadIdx.x;
    float temp;
    
    while (gid / (WARP_WIDTH*WARPS_PER_TRAJ) < n_sequences) {
        const unsigned int s = gid / (WARP_WIDTH*WARPS_PER_TRAJ);
        const float* _sequence = sequences + cum_sequence_lengths[s] * n_features;
        const unsigned int lid = gid % WARP_WIDTH;
        const unsigned int jteam = (gid % (WARP_WIDTH*WARPS_PER_TRAJ)) / WARP_WIDTH;

        for (int t = 0; t < sequence_lengths[s]; t++) {
            for (int j = jteam; j < n_states; j += WARPS_PER_TRAJ) {
                float accumulator = 0;
                for (int i = lid; i < FEATURE_WIDTH; i += WARP_WIDTH) {
                    if (i < n_features) {
                        const float mu = means[j*n_features + i];
                        const float sigma2 = variances[j*n_features + i];
                        const float x = _sequence[t*n_features + i];
                        temp = -0.5f*(log_M_2_PI + log(sigma2) + (x-mu)*(x-mu)/sigma2);
                    } else
                        temp = 0;    
                    accumulator += sum<32>(temp);
                }
                if (lid == 0) {
                    loglikelihoods[cum_sequence_lengths[s]*n_states + t*n_states + j] = accumulator;
               }
            }
        }
        gid += gridDim.x*blockDim.x;
    }
}
