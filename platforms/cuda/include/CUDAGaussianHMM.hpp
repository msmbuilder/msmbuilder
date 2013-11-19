#ifndef MIXTAPE_CUDAGAUSSIANHMM_H
#define MIXTAPE_CUDAGAUSSIANHMM_H
#include <stdlib.h>
#include <vector>
namespace Mixtape {

class CUDAGaussianHMM {
public:
    CUDAGaussianHMM(const float* trajectories,
                    const int n_trajectories,
                    const int* n_observations,
                    const int n_states,
                    const int n_features);
    void setMeans(const float* means);
    void setVariances(const float* variances);
    void setTransmat(const float* transmat);
    void setStartProb(const float* startProb);
    float computeEStep(void);
    void initializeSufficientStatistics(void);
    void computeSufficientStatistics(void);
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
    float* d_log_startprob_;
    int* d_n_observations_;
    int* d_trj_offset_;
    float* d_trajectories_;
    float* d_trajectories2_;
    float* d_ones_;
    float* d_post_;
    float* d_obs_;
    float* d_obs_squared_;
    float* d_transcounts_;
    void* cublas_handle_;


    std::vector<int> n_observations_;
    std::vector<int> trj_offset_;
    const float* trajectories_;
    int n_total_observations_;
    const int n_trajectories_;
    const int n_states_;
    const int n_features_;
};

}  //namespace

#endif
