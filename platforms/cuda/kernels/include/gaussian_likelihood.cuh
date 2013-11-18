#ifndef MIXTAPE_CUDA_GAUSSIAN_LIKELIHOOD_H
#define MIXTAPE_CUDA_GAUSSIAN_LIKELIHOOD_H
#include <stdlib.h>
#ifdef __cplusplus
extern "C" {
#endif

__global__ void gaussian_likelihood(
const float* __restrict__ sequences,
const float* __restrict__ means,
const float* __restrict__ variances,
const size_t* __restrict__ n_observations,
const size_t* __restrict__ trj_offsets,
const size_t n_trajs,
const size_t n_states,
const size_t n_features,
float* __restrict__ loglikelihoods);


#ifdef __cplusplus
}
#endif
#endif