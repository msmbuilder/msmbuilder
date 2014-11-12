cdef extern from "ghmm_estep.hpp" namespace "Mixtape":
    void do_estep_single "Mixtape::do_ghmm_estep<float>"(
        const float* log_transmat, const float* log_transmat_T,
        const float* log_startprob, const float* means,
        const float* variances, const float** sequences,
        const int n_sequences, const int* sequence_lengths,
        const int n_features, const int n_states,
        float* transcounts, float* obs, float* obs2,
        float* post, float* logprob) nogil
    void do_estep_mixed "Mixtape::do_ghmm_estep<double>"(
        const float* log_transmat, const float* log_transmat_T,
        const float* log_startprob, const float* means,
        const float* variances, const float** sequences,
        const int n_sequences, const int* sequence_lengths,
        const int n_features, const int n_states,
        float* transcounts, float* obs, float* obs2,
        float* post, float* logprob) nogil

cdef extern from "gaussian_likelihood.h":
     void gaussian_loglikelihood_diag(const float* sequence,
                                 const float*  sequence2,
                                 const float* means,
                                 const float* variances,
                                 const float* means_over_variances,
                                 const float* means2_over_variances,
                                 const float* log_variances,
                                 const int n_observations,
                                 const int n_states, const int n_features,
                                 float* loglikelihoods)

     void gaussian_loglikelihood_full(const float*  sequence,
                                 const float*  means,
                                 const float*  covariances,
                                 const int n_observations,
                                 const int n_states,
                                 const int n_features,
                                 float*  loglikelihoods)
