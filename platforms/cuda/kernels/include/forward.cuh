#ifndef MIXTAPE_CUDA_FORWARD_H
#define MIXTAPE_CUDA_FORWARD_H
#include <stdlib.h>
#ifdef __cplusplus
extern "C" {
#endif

__global__ void do_forward(
const float* __restrict__ log_transmat_T,
const float* __restrict__ log_startprob,
const float* __restrict__ frame_logprob,
const size_t n_trajs,
const size_t* __restrict__ trj_offset,
const size_t* __restrict__ n_observations,
const size_t n_states,
float* __restrict__ fwdlattice);

#ifdef __cplusplus
}
#endif
#endif