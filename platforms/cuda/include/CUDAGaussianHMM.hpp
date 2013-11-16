#ifndef MIXTAPE_CUDAGAUSSIANHMM_H
#define MIXTAPE_CUDAGAUSSIANHMM_H
#include <stdlib.h>
#include <stdio.h>
#include "utils.h"
#include "math.h"
#include <vector>
namespace Mixtape {

#define CudaSafeCall(err) __cudaSafeCall(err, __FILE__, __LINE__)
inline void __cudaSafeCall( cudaError err, const char *file, const int line) {
    if (cudaSuccess != err) {
        fprintf(stderr, "cudaSafeCall() failed at %s:%i : %s\n",
                file, line, cudaGetErrorString(err));
            exit(EXIT_FAILURE);
    }
    return;
}


class CUDAGaussianHMM {
public:
    CUDAGaussianHMM(const float* trajectories,
                   const size_t n_trajectories,
                   const size_t* n_observations,
                   const size_t n_states,
                   const size_t n_features)
    : trajectories_(trajectories)
    , n_trajectories_(n_trajectories)
    , n_total_observations_(0)
    , n_states_(n_states)
    , n_features_(n_features)
    , d_fwdlattice_(NULL)
    , d_bwdlattice_(NULL)
    , d_posteriors_(NULL)
    , d_framelogprob_(NULL)
    , d_log_transmat_(NULL)
    , d_log_transmat_T_(NULL)
    , d_means_(NULL)
    , d_variances_(NULL)
    , d_log_startprob_(NULL)
    , d_n_observations_(NULL)
    , d_trj_offset_(NULL)
    , d_trajectories_(NULL)
    {
        n_total_observations_ = 0;
        n_observations_.resize(n_trajectories);
        for (int s = 0; s < n_trajectories_; s++) {
            n_total_observations_ += n_observations[s];
            n_observations_[s] = n_observations[s];
        }

        // trj_offset_[s] is the index in the trajectories_ memory
        // blob where the s-th trajectory starts
        trj_offset_[0] = 0;
        trj_offset_.resize(n_trajectories);
        for (int s = 1; s < n_trajectories_; s++)
            trj_offset_[s] = trj_offset_[s-1] + n_observations_[s-1]*n_features;

        CudaSafeCall(cudaMalloc((void **) &d_trajectories_, n_total_observations_*n_features_*sizeof(float)));
        CudaSafeCall(cudaMalloc((void **) &d_fwdlattice_, n_total_observations_*n_states_*sizeof(float)));
        CudaSafeCall(cudaMalloc((void **) &d_bwdlattice_, n_total_observations_*n_states_*sizeof(float)));
        CudaSafeCall(cudaMalloc((void **) &d_posteriors_, n_total_observations_*n_states_*sizeof(float)));
        CudaSafeCall(cudaMalloc((void **) &d_framelogprob_, n_states_*n_states_*sizeof(float)));
        CudaSafeCall(cudaMalloc((void **) &d_log_transmat_, n_states_*n_states_*sizeof(float)));
        CudaSafeCall(cudaMalloc((void **) &d_log_transmat_T_, n_states_*n_states_*sizeof(float)));
        CudaSafeCall(cudaMalloc((void **) &d_means_, n_states_*n_features_*sizeof(float)));
        CudaSafeCall(cudaMalloc((void **) &d_variances_, n_states_*n_features_*sizeof(float)));
        CudaSafeCall(cudaMalloc((void **) &d_log_startprob_, n_states_*sizeof(float)));
        CudaSafeCall(cudaMalloc((void **) &d_n_observations_, n_trajectories_*sizeof(float)));
        CudaSafeCall(cudaMalloc((void **) &d_trj_offset_, n_trajectories_*sizeof(float)));

        CudaSafeCall(cudaMemcpy(d_trajectories_, trajectories_, n_total_observations_*n_features_*sizeof(float), cudaMemcpyHostToDevice));
        CudaSafeCall(cudaMemcpy(d_n_observations_, &n_observations_[0], n_trajectories_*sizeof(float), cudaMemcpyHostToDevice));
        CudaSafeCall(cudaMemcpy(d_trj_offset_, &trj_offset_[0], n_trajectories_*sizeof(float), cudaMemcpyHostToDevice));
    }

    void setMeans(const float* means) {
        CudaSafeCall(cudaMemcpy(d_means_, means, n_states_*n_features_*sizeof(float), cudaMemcpyHostToDevice));
    }

    void setVariances(const float* variances) {
        CudaSafeCall(cudaMemcpy(d_variances_, variances, n_states_*n_features_*sizeof(float), cudaMemcpyHostToDevice));
    }

    void setTransmat(const float* transmat) {
        std::vector<float> log_transmat(n_states_*n_states_);
        std::vector<float> log_transmat_T(n_states_*n_states_);
        for (int i = 0; i < n_states_; i++)
            for (int j = 0; j < n_states_; j++) {
                log_transmat[i*n_states_ + j] = log(transmat[i*n_states_ + j]);
                log_transmat_T[j*n_states_ +i] = log_transmat[i*n_states_ + j];
            }
        CudaSafeCall(cudaMemcpy(d_log_transmat_, &log_transmat[0],     n_states_*n_states_*sizeof(float), cudaMemcpyHostToDevice));
        CudaSafeCall(cudaMemcpy(d_log_transmat_T_, &log_transmat_T[0], n_states_*n_states_*sizeof(float), cudaMemcpyHostToDevice));
    }

    void setStartProb(const float* startProb) {
        std::vector<float> log_startprob(n_states_);
        for (int i = 0; i < n_states_; i++)
            log_startprob[i] = log(startProb[i]);
        CudaSafeCall(cudaMemcpy(d_log_startprob_, &log_startprob[0], n_states_*sizeof(float), cudaMemcpyHostToDevice));
    }

    float doEStep(void);

    
    ~CUDAGaussianHMM() {
        CudaSafeCall(cudaFree(d_trajectories_));
        CudaSafeCall(cudaFree(d_fwdlattice_));
        CudaSafeCall(cudaFree(d_bwdlattice_));
        CudaSafeCall(cudaFree(d_posteriors_));
        CudaSafeCall(cudaFree(d_framelogprob_));
        CudaSafeCall(cudaFree(d_log_transmat_));
        CudaSafeCall(cudaFree(d_log_transmat_T_));
        CudaSafeCall(cudaFree(d_means_));
        CudaSafeCall(cudaFree(d_variances_));
        CudaSafeCall(cudaFree(d_log_startprob_));
        CudaSafeCall(cudaFree(d_n_observations_));
        CudaSafeCall(cudaFree(d_trj_offset_));
    }


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
    size_t* d_n_observations_;
    size_t* d_trj_offset_;
    float* d_trajectories_;

    std::vector<size_t> n_observations_;
    std::vector<size_t> trj_offset_;
    const float* trajectories_;
    size_t n_total_observations_;
    const size_t n_trajectories_;
    const size_t n_states_;
    const size_t n_features_;
};

}  //namespace

#endif
