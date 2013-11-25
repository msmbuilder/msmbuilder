/*****************************************************************/
/*    Copyright (c) 2013, Stanford University and the Authors    */
/*    Author: Robert McGibbon <rmcgibbo@gmail.com>               */
/*    Contributors:                                              */
/*                                                               */
/*****************************************************************/

#ifndef MIXTAPE_CUDAGAUSSIANHMM_H
#define MIXTAPE_CUDAGAUSSIANHMM_H
#include <stdlib.h>
#include <vector>
namespace Mixtape {

class CUDAGaussianHMM {
public:
    CUDAGaussianHMM(const int n_states,
                    const int n_features);
    void setSequences(const float** sequences,
                      const int n_sequences,
                      const int* sequence_lengths);
    void delSequences();
    void setMeans(const float* means);
    void getMeans(float* out);
    void setVariances(const float* variances);
    void getVariances(float* out);
    void setTransmat(const float* transmat);
    void getTransmat(float* out);
    void setStartProb(const float* startProb);
    void getStartProb(float* out);
    float computeEStep(void);
    void initializeSufficientStatistics(void);
    float computeSufficientStatistics(void);
    void getFrameLogProb(float* out);
    void getFwdLattice(float* out);
    void getBwdLattice(float* out);
    void getPosteriors(float* out);

    void getStatsObs(float* out);
    void getStatsObsSquared(float* out);
    void getStatsPost(float* out);
    void getStatsTransCounts(float* out);
    ~CUDAGaussianHMM();


private:
    float* d_fwdlattice_;
    float* d_bwdlattice_;
    float* d_posteriors_;
    float* d_framelogprob_;
    float* d_log_transmat_;
    float* d_log_transmat_T_;
    float* d_means_;
    float* d_variances_;
    float* d_logvariances_;
    float* d_log_startprob_;
    float* d_logprob_;
    int*   d_sequence_lengths_;
    int*   d_cum_sequence_lengths_;
    float* d_sequences_;
    float* d_sequences2_;
    float* d_ones_;
    float* d_post_;
    float* d_obs_;
    float* d_obs_squared_;
    float* d_transcounts_;
    void* cublas_handle_;

    int n_observations_;
    int n_sequences_;
    std::vector<int> sequence_lengths_;
    std::vector<int> cum_sequence_lengths_;
    std::vector<int> trj_offset_;
    const int n_states_;    // true number of states
    const int n_pstates_;   // number of "padded" states
    const int n_features_;
};

}  //namespace

#endif
