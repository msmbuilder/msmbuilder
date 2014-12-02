#ifndef MIXTAPE_CPU_GAUSSIAN_LIKELIHOOD_H
#define MIXTAPE_CPU_GAUSSIAN_LIKELIHOOD_H
#ifdef __cplusplus
extern "C" {
#endif

void gaussian_loglikelihood_diag(const float* __restrict sequence,
                                 const float* __restrict sequence2,
                                 const float* __restrict means,
                                 const float* __restrict variances,
                                 const float* __restrict means_over_variances,
                                 const float* __restrict means2_over_variances,
                                 const float* __restrict log_variances,
                                 const int n_observations,
                                 const int n_states, const int n_features,
                                 float* __restrict loglikelihoods);

void gaussian_loglikelihood_full(const float* __restrict sequence,
                                 const float* __restrict means,
                                 const float* __restrict covariances,
                                 const int n_observations,
                                 const int n_states,
                                 const int n_features,
                                 float* __restrict loglikelihoods);

void gaussian_lds_loglikelihood_full(const float* __restrict sequence,
                                 const float* __restrict As,
                                 const float* __restrict bs,
                                 const float* __restrict Qs,
                                 const int n_observations,
                                 const int n_states,
                                 const int n_features,
                                 float* __restrict lds_loglikelihoods);
#ifdef __cplusplus
}
#endif
#endif
