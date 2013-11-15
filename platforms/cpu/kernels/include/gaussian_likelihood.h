#ifndef MIXTAPE_GAUSSIAN_LIKELIHOOD_H
#define MIXTAPE_GAUSSIAN_LIKELIHOOD_H
#include <stdlib.h>

void gaussian_loglikelihood_diag(const float* __restrict__ sequences, const float* __restrict__ means,
                                 const float* __restrict__ variances, const size_t n_trajs,
                                 const size_t* __restrict__ n_observations, const size_t n_states, const size_t n_features,
                                     float* __restrict__ loglikelihoods);

#endif
