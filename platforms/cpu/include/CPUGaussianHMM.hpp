#include <stdlib.h>
#include "math.h"
#include <vector>

namespace Mixtape {

class CPUGaussianHMM {
public:
    CPUGaussianHMM(const float* trajectories,
                   const size_t n_trajectories,
                   const size_t* n_observations,
                   const size_t n_states,
                   const size_t n_features)
    : trajectories_(trajectories)
    , n_trajectories_(n_trajectories)
    , n_states_(n_states)
    , n_features_(n_features)
    , means_(NULL)
    , variances_(NULL)
    {
        n_total_observations_ = 0;
        n_observations_.resize(n_trajectories);
        for (int s = 0; s < n_trajectories_; s++) {
            n_total_observations_ += n_observations_[s];
            n_observations_[s] = n_observations[s];
        }

        fwdlattice_.resize(n_total_observations_ * n_states_);
        bwdlattice_.resize(n_total_observations_ * n_states_);
        posteriors_.resize(n_total_observations_ * n_states_);
        framelogprob_.resize(n_total_observations_ * n_states_);


    }

    void setMeans(float* means) {
        means_.resize(n_states_ * n_features_);
        for (int i = 0; i < n_states_; i++)
            for (int j = 0; j < n_features_; j++)
                means_[i*n_features_ + j] = means[i*n_features_ + j];
    }

    void setVariances(float* variances) {
        variances_.resize(n_states_ * n_features_);
        for (int i = 0; i < n_states_; i++)
            for (int j = 0; j < n_features_; j++)
                variances_[i*n_features_ + j] = variances[i*n_features_ + j];
    }
    void setTransmat(float* transmat) {
        log_transmat_.resize(n_states_ * n_states_);
        log_transmat_T_.resize(n_states_ * n_states_);
        for (int i = 0; i < n_states_; i++)
            for (int j = 0; j < n_states_; j++) {
                log_transmat_[i*n_states_ + j] = log(transmat[i*n_states_ + j]);
                log_transmat_T_[j*n_states_ +i] = log_transmat_[i*n_states_ + j];
            }
    }

    void setStartProb(float* startProb) {
        log_startprob_.resize(n_states_);
        for (int i = 0; i < n_states_; i++)
            log_startprob_[i] = log(startProb[i]);
    }

    float doMStep(void);
    
    ~CPUGaussianHMM() { }


private:
    std::vector<float> fwdlattice_;
    std::vector<float> bwdlattice_;
    std::vector<float> posteriors_;
    std::vector<float> framelogprob_;
    std::vector<float> log_transmat_;
    std::vector<float> log_transmat_T_;
    std::vector<float> means_;
    std::vector<float> variances_;
    std::vector<float> log_startprob_;
    std::vector<size_t> n_observations_;

    size_t n_total_observations_;
    const float* trajectories_;
    const size_t n_trajectories_;
    const size_t n_states_;
    const size_t n_features_;
};

}  //namespace
