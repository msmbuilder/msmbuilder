#ifndef MIXTAPE_LOGSUMEXP_H
#define MIXTAPE_LOGSUMEXP_H
/**
 * Compute the log of the sum of the exponential of `value` across a width-32 warp.
 * This function must be called by each thread in the warp, which each contains a
 * different `value`. The result will appear on the 0-th thread, and is stored in
 * location addressed by the result pointer. Nothing will be written into the result
 * pointer on the other threds.
 *
 * Note that the summation will only occur across a single warp.
 */
__device__ void logsumexp32(float value, float* result)
{
    const unsigned int gid = blockIdx.x*blockDim.x+threadIdx.x;
    const unsigned int lid = gid % 32;
    float max;

#if __CUDA_ARCH__ >= 300
    max = value;
    #pragma unroll
    for(int offset = 1; offset < 32; offset <<= 1)
        max = fmaxf(max, __shfl_down(max, offset));

    max = __shfl(max, 0);
    value = __expf(value - max);

    #pragma unroll
    for(int offset = 1; offset < 32; offset <<= 1)
        value += __shfl_down(value, offset);

    if (lid == 0)
        *result = logf(value) + max;
#else
    __shared__ float s[32];
    s[lid] = value;
    s[lid] = fmaxf(s[lid], s[(lid - 1) % 32]);
    s[lid] = fmaxf(s[lid], s[(lid - 2) % 32]);
    s[lid] = fmaxf(s[lid], s[(lid - 4) % 32]);
    s[lid] = fmaxf(s[lid], s[(lid - 8) % 32]);
    s[lid] = fmaxf(s[lid], s[(lid - 16) % 32]);

    max = s[0];
    s[lid] = __expf(value - max);

    s[lid] += s[(lid - 1) % 32];
    s[lid] += s[(lid - 2) % 32];
    s[lid] += s[(lid - 4) % 32];
    s[lid] += s[(lid - 8) % 32];
    s[lid] += s[(lid - 16) % 32];

    if (lid == 0)
        *result = logf(s[0]) + max;
#endif

}

#endif
