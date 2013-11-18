#ifndef MIXTAPE_CUDA_FORWARD_H
#define MIXTAPE_CUDA_FORWARD_H
#include <stdlib.h>
#ifdef __cplusplus
extern "C" {
#endif

__global__ void backward4(
const float* __restrict__ log_transmat,
const float* __restrict__ log_startprob,
const float* __restrict__ frame_logprob,
const size_t* __restrict__ n_observations,
const size_t* __restrict__ trj_offsets,
const size_t n_trajs,
float* __restrict__ bwdlattice);


#ifdef __cplusplus
}
#endif
#endif