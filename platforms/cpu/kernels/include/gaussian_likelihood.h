#ifndef MIXTAPE_CPU_GAUSSIAN_LIKELIHOOD_H
#define MIXTAPE_CPU_GAUSSIAN_LIKELIHOOD_H
#ifdef __cplusplus
extern "C" {
#endif

void gaussian_loglikelihood_diag(const float* __restrict__ sequence,
                                 const double* __restrict__ means,
                                 const double* __restrict__ variances,
                                 const double* __restrict__ means_over_variances,
                                 const double* __restrict__ means2_over_variances,
                                 const double* __restrict__ log_variances,
                                 const int n_observations,
                                 const int n_states, const int n_features,
                                 float* __restrict__ loglikelihoods);

void gaussian_loglikelihood_full(const float* __restrict__ sequence,
                                 const float* __restrict__ means,
                                 const float* __restrict__ covariances,
                                 const int n_observations,
                                 const int n_states,
                                 const int n_features,
                                 float* __restrict__ loglikelihoods);

void gaussian_lds_loglikelihood_full(const float* __restrict__ sequence,
                                 const float* __restrict__ As,
                                 const float* __restrict__ bs,
                                 const float* __restrict__ Qs,
                                 const int n_observations,
                                 const int n_states,
                                 const int n_features,
                                 float* __restrict__ lds_loglikelihoods);
#ifdef __cplusplus
}
#endif
#endif
