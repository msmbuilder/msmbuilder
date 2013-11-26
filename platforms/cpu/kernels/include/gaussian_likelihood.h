#ifndef MIXTAPE_CPU_GAUSSIAN_LIKELIHOOD_H
#define MIXTAPE_CPU_GAUSSIAN_LIKELIHOOD_H
#ifdef __cplusplus
extern "C" {
#endif

void gaussian_loglikelihood_diag(const float* __restrict__ sequence,
                                 const float* __restrict__ sequence2,
                                 const float* __restrict__ means,
                                 const float* __restrict__ variances,
                                 const float* __restrict__ means_over_variances,
                                 const float* __restrict__ means2_over_variances,
                                 const float* __restrict__ log_variances,
                                 const int n_observations,
                                 const int n_states, const int n_features,
                                 float* __restrict__ loglikelihoods);

#ifdef __cplusplus
}
#endif
#endif
