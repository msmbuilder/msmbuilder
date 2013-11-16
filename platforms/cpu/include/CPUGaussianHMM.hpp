#ifndef MIXTAPE_CPUGAUSSIANHMM_H
#define MIXTAPE_CPUGAUSSIANHMM_H
#include "utils.h"
#include <stdlib.h>
#include "math.h"
#include <vector>
namespace Mixtape {

class CPUGaussianHMM {
public:
    /**
     * Create a new CPUGaussianHMM object. The CPUGaussianHMM implemenets the compute
     * intensive portion of the EM algorithm for HMMs, which is the E step and
     * computing sufficient statistics for the M step.
     * 
     * Trajectory data -- a concatenation of 2D trajectories of shape
     * [n_observations, n_features] where each trajectory may be of different length --
     * is passed in by reference and is NOT owned or necessarily copied by this class.
     * The caller must ensure that the trajectory data isn't garbage collected while this
     * class is using it.
     *
     * All other data (means, variances, etc) is much smaller, and is copied into internal
     * data structures whose memory is owned and manager by this class
     */
    CPUGaussianHMM(const float* trajectories,
                   const size_t n_trajectories,
                   const size_t* n_observations,
                   const size_t n_states,
                   const size_t n_features)
    : trajectories_(trajectories)
    , n_trajectories_(n_trajectories)
    , n_states_(n_states)
    , n_features_(n_features)
    {
        n_total_observations_ = 0;
        n_observations_.resize(n_trajectories);
        for (int s = 0; s < n_trajectories_; s++) {
            n_total_observations_ += n_observations[s];
            n_observations_[s] = n_observations[s];
        }
        fwdlattice_.resize(n_total_observations_ * n_states_);
        bwdlattice_.resize(n_total_observations_ * n_states_);
        posteriors_.resize(n_total_observations_ * n_states_);
        framelogprob_.resize(n_total_observations_ * n_states_);
    }

    void setMeans(const float* means) {
        means_.resize(n_states_ * n_features_);
        for (int i = 0; i < n_states_; i++)
            for (int j = 0; j < n_features_; j++)
                means_[i*n_features_ + j] = means[i*n_features_ + j];
    }

    void setVariances(const float* variances) {
        variances_.resize(n_states_ * n_features_);
        for (int i = 0; i < n_states_; i++)
            for (int j = 0; j < n_features_; j++)
                variances_[i*n_features_ + j] = variances[i*n_features_ + j];
    }
    void setTransmat(const float* transmat) {
        log_transmat_.resize(n_states_ * n_states_);
        log_transmat_T_.resize(n_states_ * n_states_);
        for (int i = 0; i < n_states_; i++)
            for (int j = 0; j < n_states_; j++) {
                log_transmat_[i*n_states_ + j] = log(transmat[i*n_states_ + j]);
                log_transmat_T_[j*n_states_ +i] = log_transmat_[i*n_states_ + j];
            }
    }

    void setStartProb(const float* startProb) {
        log_startprob_.resize(n_states_);
        for (int i = 0; i < n_states_; i++)
            log_startprob_[i] = log(startProb[i]);
    }

    float doEStep(void);

    void printFwdLattice() {
    printf("fwdlattice\n"); pprintarray(&fwdlattice_[0], n_observations_[0], n_states_);
    }

    void printFrameLogprob() {
    printf("framelogprob\n"); pprintarray(&framelogprob_[0], n_observations_[0], n_states_);
    }

    void printBwdLattice() {
        printf("bwdlattice\n"); pprintarray(&bwdlattice_[0], n_observations_[0], n_states_);
    }

    void printPosteriors() {
    printf("posteriors\n"); pprintarray(&posteriors_[0], n_observations_[0], n_states_);
    }
    
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

#endif
