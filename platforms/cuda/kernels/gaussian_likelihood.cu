/*****************************************************************/
/*    Copyright (c) 2013, Stanford University and the Authors    */
/*    Author: Robert McGibbon <rmcgibbo@gmail.com>               */
/*    Contributors:                                              */
/*                                                               */
/*****************************************************************/

#include "logsumexp.cu"
#include "staticassert.cuh"
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


template <const unsigned int W1, const unsigned int W2>
__global__ void log_diag_mvn_likelihood(
const float* __restrict__ sequences,
const float* __restrict__ means,
const float* __restrict__ variances,
const float* __restrict__ logvariances,
const int n_samples,
const int n_states,
const int n_features,
float* __restrict__ loglikelihoods)
{
    /* W1 and W2 are the two chunk size parameters that control the dimensions
     * of the submatrices that are loaded into shared memory. The product W1*W2
     * currently must be equal to the size of the block that the kernel is
     * invoked with.
     *
     * From `sequences`, submatrices with W1 samples and W2 features are loaded,
     * and from means/variances/logvariances, submatrices with W1 states and
     * W2 features are loaded. In general, bigger blocks are more efficient but
     * only for bigger matrices, because spillover, where, for example,
     * n_features % W2 != 0 causes some inefficiency.
     *
     *
     * This kernel is optimized for n_samples >> (n_states ~ n_features) because
     * it deploys all of its threads along the n_samples dimension, and each thread
     * has to iterate in the blocked n_states/n_featues space, computing a
     * total of (n_states/W1) * (n_features/W2) entries per thread gid iteration.
     */
    __shared__ float SEQ[W1][W2];
    __shared__ float MU[W1][W2];
    __shared__ float SIG2[W1][W2];
    __shared__ float LOGSIG2[W1][W2];
    unsigned int gid = blockIdx.x*blockDim.x+threadIdx.x;
    const float MINUS_HALF_LOG_2_PI = -0.91893853320467267f;

    while (gid/(W1*W2) < (n_samples+W1-1)/W1) {
        const unsigned int lid = gid % (W1*W2);
        const unsigned int loadRow = lid / W2;
        const unsigned int loadCol = lid % W2;
        const unsigned int samplesBlock = W1*(gid / (W1*W2));
        const bool validSample = (samplesBlock+loadRow) < n_samples;

        for (unsigned int featuresBlock = 0; featuresBlock < ((n_features+W2-1)/W2)*W2; featuresBlock += W2) {
            const bool validFeature = (featuresBlock+loadCol) < n_features;
            SEQ[loadRow][loadCol] = (validSample && validFeature) ? sequences[(samplesBlock+loadRow)*n_features + (featuresBlock+loadCol)] : 0.0f;
        for (unsigned int statesBlock = 0; statesBlock < ((n_states+W1-1)/W1)*W1; statesBlock += W1) {
            // Load 64 W1*W2 items into shared memory from the global arrays
            const bool validState = (statesBlock+loadRow) < n_states;
            if (validState && validFeature) {
                MU[loadRow][loadCol] = means[(statesBlock+loadRow)*n_features + (featuresBlock+loadCol)];
                SIG2[loadRow][loadCol] = variances[(statesBlock+loadRow)*n_features + (featuresBlock+loadCol)];
                LOGSIG2[loadRow][loadCol] = logvariances[(statesBlock+loadRow)*n_features + (featuresBlock+loadCol)];
            } else {
                // Use sig < 0 as a sentinel for the block containing an out of bound
                // index, which is necessary for the boundary blocks without making an
                // explicit epilogue to handle them.
                MU[loadRow][loadCol] = 0;
                SIG2[loadRow][loadCol] = -1;
                LOGSIG2[loadRow][loadCol] = 0;
            }
            __syncthreads();


            // Now, we need to compute W1^2 results, each of which is a sum of W2 entries
            // between the W1*W2 threads on this. So each thread needs to do W1 entries
            // in the sum.

            // Say W1=2, W2=32. There are four sums of 32 elements to compute. Each
            // sum is split up into W2/W1=16 chunks of 2 elements each.

            // thread 0 takes (0, 0) from 0 to W1
            // thread 1 takes (0, 0) from W1 to 2*W1
            // ...
            // thread   takes (0, 0) from .. to W2
            // ...
            // takes (W1-1, W1-1) from W2-W1 to W2

            const unsigned int offset = lid % (W2 / W1);
            const unsigned int i = (lid / (W2/W1)) % W1;
            const unsigned int j = (lid / (W2/W1)) / W1;
            float temp = 0;
            #pragma unroll
            for (int k = W1*offset; k < W1*(offset+1); k++) {
                float f = SEQ[i][k] - MU[j][k];
                float loglike = MINUS_HALF_LOG_2_PI + -0.5f * (LOGSIG2[j][k] + (f*f / SIG2[j][k]));
                // using the sentinel to avoid summing in invalid indices
                temp += loglike * (SIG2[j][k] > 0);
        }
            atomicAdd(loglikelihoods + (samplesBlock+i)*n_states + statesBlock+j, temp);
        } }
        gid += gridDim.x*blockDim.x;
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
