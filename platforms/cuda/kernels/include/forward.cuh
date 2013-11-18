#ifndef MIXTAPE_CUDA_FORWARD_H
#define MIXTAPE_CUDA_FORWARD_H
#include <stdlib.h>
#ifdef __cplusplus
extern "C" {
#endif

__global__ void forward4(
const float* __restrict__ log_transmat_T,
const float* __restrict__ log_startprob,
const float* __restrict__ frame_logprob,
const size_t* __restrict__ n_observations,
const size_t* __restrict__ trj_offsets,
const size_t n_trajs,
float* __restrict__ fwdlattice);


__global__ void forward8(
const float* __restrict__ log_transmat_T,
const float* __restrict__ log_startprob,
const float* __restrict__ frame_logprob,
const size_t* __restrict__ n_observations,
const size_t* __restrict__ trj_offsets,
const size_t n_trajs,
float* __restrict__ fwdlattice);


__global__ void forward16(
const float* __restrict__ log_transmat_T,
const float* __restrict__ log_startprob,
const float* __restrict__ frame_logprob,
const size_t* __restrict__ n_observations,
const size_t* __restrict__ trj_offsets,
const size_t n_trajs,
float* __restrict__ fwdlattice);


__global__ void forward32(
const float* __restrict__ log_transmat_T,
const float* __restrict__ log_startprob,
const float* __restrict__ frame_logprob,
const size_t* __restrict__ n_observations,
const size_t* __restrict__ trj_offsets,
const size_t n_trajs,
float* __restrict__ fwdlattice);


#ifdef __cplusplus
}
#endif
#endif