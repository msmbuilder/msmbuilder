#ifndef MIXTAPE_CUDA_POSTERIORS_H
#define MIXTAPE_CUDA_POSTERIORS_H
#include <stdlib.h>
#ifdef __cplusplus
extern "C" {
#endif

__global__ void posteriors4(
const float* __restrict__ fwdlattice,
const float* __restrict__ bwdlattice,
const size_t* __restrict__ n_observations,
const size_t* __restrict__ trj_offsets,
const size_t n_trajs,
float* __restrict__ posteriors);


#ifdef __cplusplus
}
#endif
#endif