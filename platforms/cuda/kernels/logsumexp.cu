/*****************************************************************/
/*    Copyright (c) 2013, Stanford University and the Authors    */
/*    Author: Robert McGibbon <rmcgibbo@gmail.com>               */
/*    Contributors:                                              */
/*                                                               */
/*****************************************************************/

#ifndef MIXTAPE_LOGSUMEXP_H
#define MIXTAPE_LOGSUMEXP_H


/**
 * Compute the log of the sum of the exponential of `value` across a width-32 warp.
 * This function must be called by each thread in the warp, which each contains a
 * different `value`.
 *
 * The template parameter N needsto be a power of 2. When 32, the summation
 * goes accross the entire warp. When 16, the first 16 threads do one summation
 * and the second 16 threads to a different sum. Etc.
 *
 */

template <int N>
__device__ float logsumexp(float value)
{
    float max = value;
#if __CUDA_ARCH__ >= 300
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
#else
    return 0;
    // const unsigned int gid = blockIdx.x*blockDim.x+threadIdx.x;
    // const unsigned int lid = gid % 32;
    // __shared__ volatile float s[32];
    // 
    // s[lid] = value;
    // for(int offset = 1; offset < N; offset <<= 1)
    //     s[lid] = fmaxf(s[lid], s[(lid - offset) % 32]);
    // max = s[lid / N];
    // s[lid] = __expf(value - max);
    // 
    // for(int offset = 1; offset < N; offset <<= 1)
    //     s[lid] += s[(lid - offset) % 32];
    // 
    // s[lid] = logf(value) + max;
    // for(int offset = 1; offset < N; offset <<= 1)
    //     s[lid] = s[(lid + offset) % 32];
    // 
    // return s[lid];
#endif
}

template <int N>
__device__ float sum(float value) {
#if __CUDA_ARCH__ >= 300
    for(int offset = 1; offset < N; offset <<= 1)
        value += __shfl_down(value, offset);
#else
    const unsigned int gid = blockIdx.x*blockDim.x+threadIdx.x;
    const unsigned int lid = gid % 32;
    __shared__ volatile float s[32];
    s[lid] = value;

    for(int offset = 1; offset < N; offset <<= 1)
        s[lid] += s[(lid + offset) % N];
    value = s[lid];
#endif
    return value;
}


/**
 * Round up to next higher power of 2 (return x if it's already a power
 */
__device__ inline int pow2roundup(int x)
{
    if (x < 0)
        return 0;
    --x;
    x |= x >> 1;
    x |= x >> 2;
    x |= x >> 4;
    x |= x >> 8;
    x |= x >> 16;
    return x+1;
}

#endif
