#include <stdlib.h>
#include <stdio.h>
#include <vector>
#include <string>
#include "CUDAGaussianHMM.hpp"

#include "forward.cu"
#include "backward.cu"
#include "gaussian_likelihood.cu"
#include "posteriors.cu"


#include <cublas_v2.h>

#define CudaSafeCall(err) __cudaSafeCall(err, __FILE__, __LINE__)
#define CudaCheckError()  __cudaCheckError( __FILE__, __LINE__ )

inline void __cudaSafeCall( cudaError err, const char *file, const int line) {
    if (cudaSuccess != err) {
        fprintf(stderr, "cudaSafeCall() failed at %s:%i : %s\n",
                file, line, cudaGetErrorString(err));
            exit(EXIT_FAILURE);
    }
    return;
}
inline void __cudaCheckError( const char *file, const int line ) {
#ifdef CUDA_ERROR_CHECK
    cudaError err = cudaGetLastError();
    if (cudaSuccess != err) {
        fprintf(stderr, "cudaCheckError() failed at %s:%i : %s\n",
                file, line, cudaGetErrorString(err));
        exit(-1);
    }
 
    // More careful checking. However, this will affect performance.
    // Comment away if needed.
    err = cudaDeviceSynchronize();
    if(cudaSuccess != err) {
        fprintf(stderr, "cudaCheckError() with sync failed at %s:%i : %s\n",
                file, line, cudaGetErrorString(err));
        exit(-1);
    }
#endif
 
    return;
}


namespace Mixtape {
CUDAGaussianHMM::CUDAGaussianHMM(const float* trajectories,
                                 const int n_trajectories,
                                 const int* n_observations,
                                 const int n_states,
                                 const int n_features)
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
    trj_offset_.resize(n_trajectories);
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
    CudaSafeCall(cudaMalloc((void **) &d_trajectories2_, n_total_observations_*n_features_*sizeof(float)));
    CudaSafeCall(cudaMalloc((void **) &d_fwdlattice_, n_total_observations_*n_states_*sizeof(float)));
    CudaSafeCall(cudaMalloc((void **) &d_bwdlattice_, n_total_observations_*n_states_*sizeof(float)));
    CudaSafeCall(cudaMalloc((void **) &d_posteriors_, n_total_observations_*n_states_*sizeof(float)));
    CudaSafeCall(cudaMalloc((void **) &d_framelogprob_, n_total_observations_*n_states_*sizeof(float)));
    CudaSafeCall(cudaMalloc((void **) &d_log_transmat_, n_states_*n_states_*sizeof(float)));
    CudaSafeCall(cudaMalloc((void **) &d_log_transmat_T_, n_states_*n_states_*sizeof(float)));
    CudaSafeCall(cudaMalloc((void **) &d_means_, n_states_*n_features_*sizeof(float)));
    CudaSafeCall(cudaMalloc((void **) &d_variances_, n_states_*n_features_*sizeof(float)));
    CudaSafeCall(cudaMalloc((void **) &d_log_startprob_, n_states_*sizeof(float)));
    CudaSafeCall(cudaMalloc((void **) &d_n_observations_, n_trajectories_*sizeof(float)));
    CudaSafeCall(cudaMalloc((void **) &d_trj_offset_, n_trajectories_*sizeof(float)));
    CudaSafeCall(cudaMalloc((void **) &d_ones_,  n_total_observations_*sizeof(float)));
    
    CudaSafeCall(cudaMalloc((void **) &d_post_, n_states_*sizeof(float)));
    CudaSafeCall(cudaMalloc((void **) &d_obs_, n_states_*n_features_*sizeof(float)));
    CudaSafeCall(cudaMalloc((void **) &d_obs_squared_, n_states_*n_features_*sizeof(float)));
    CudaSafeCall(cudaMalloc((void **) &d_counts_, n_states_*n_states_*sizeof(float)));
    
    CudaSafeCall(cudaMemcpy(d_trajectories_, trajectories_, n_total_observations_*n_features_*sizeof(float), cudaMemcpyHostToDevice));
    CudaSafeCall(cudaMemcpy(d_n_observations_, &n_observations_[0], n_trajectories_*sizeof(float), cudaMemcpyHostToDevice));
    CudaSafeCall(cudaMemcpy(d_trj_offset_, &trj_offset_[0], n_trajectories_*sizeof(float), cudaMemcpyHostToDevice));

    cublasStatus_t status = cublasCreate((cublasHandle_t*) &cublas_handle_);
    if (status != CUBLAS_STATUS_SUCCESS) {
        fprintf(stderr, "!!!! CUBLAS initialization error\n");
    }

    // square the observed trajectories.
    square<<<1, 256>>>(d_trajectories_, n_total_observations_*n_features, d_trajectories2_);
    fill<<<1, 256>>>(d_ones_, 1.0, n_total_observations_);

}


float CUDAGaussianHMM::computeEStep() {
    gaussian_likelihood<<<1, 32>>>(
        d_trajectories_, d_means_, d_variances_, n_trajectories_,
        d_n_observations_, d_trj_offset_, n_states_, n_features_,
        d_framelogprob_);
    forward4<<<1, 32>>>(
        d_log_transmat_T_, d_log_startprob_, d_framelogprob_,
        d_n_observations_, d_trj_offset_, n_trajectories_,
        d_fwdlattice_);
    backward4<<<1, 32>>>(
        d_log_transmat_T_, d_log_startprob_, d_framelogprob_,
        d_n_observations_, d_trj_offset_, n_trajectories_,
        d_bwdlattice_);
    posteriors4<<<1, 32>>>(
        d_fwdlattice_, d_bwdlattice_, n_trajectories_,
        d_n_observations_, d_trj_offset_, d_posteriors_);

    CudaCheckError();
    return 1.0;
}

void CUDAGaussianHMM::setMeans(const float* means) {
    CudaSafeCall(cudaMemcpy(d_means_, means, n_states_*n_features_*sizeof(float), cudaMemcpyHostToDevice));
}

void CUDAGaussianHMM::setVariances(const float* variances) {
    CudaSafeCall(cudaMemcpy(d_variances_, variances, n_states_*n_features_*sizeof(float), cudaMemcpyHostToDevice));
}

void CUDAGaussianHMM::setTransmat(const float* transmat) {
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

void CUDAGaussianHMM::setStartProb(const float* startProb) {
    std::vector<float> log_startprob(n_states_);
    for (int i = 0; i < n_states_; i++)
        log_startprob[i] = log(startProb[i]);
}

void CUDAGaussianHMM::getFrameLogProb(float* out) {
    CudaSafeCall(cudaMemcpy(out, d_framelogprob_, n_total_observations_*n_states_*sizeof(float), cudaMemcpyDeviceToHost));
}

void CUDAGaussianHMM::getFwdLattice(float* out) {
    CudaSafeCall(cudaMemcpy(out, d_fwdlattice_, n_total_observations_*n_states_*sizeof(float), cudaMemcpyDeviceToHost));
}

void CUDAGaussianHMM::getBwdLattice(float* out) {
    CudaSafeCall(cudaMemcpy(out, d_bwdlattice_, n_total_observations_*n_states_*sizeof(float), cudaMemcpyDeviceToHost));
}

void CUDAGaussianHMM::getPosteriors(float* out) {
    CudaSafeCall(cudaMemcpy(out, d_posteriors_, n_total_observations_*n_states_*sizeof(float), cudaMemcpyDeviceToHost));
}

void CUDAGaussianHMM::getStatsObs(float* out) {
    CudaSafeCall(cudaMemcpy(out, d_obs_, n_states_*n_features_*sizeof(float), cudaMemcpyDeviceToHost));
}

void CUDAGaussianHMM::getStatsObsSquared(float* out) {
    CudaSafeCall(cudaMemcpy(out, d_obs_squared_, n_states_*n_features_*sizeof(float), cudaMemcpyDeviceToHost));
}

void CUDAGaussianHMM::getStatsPost(float* out) {
    CudaSafeCall(cudaMemcpy(out, d_post_, n_states_*sizeof(float), cudaMemcpyDeviceToHost));
}

void CUDAGaussianHMM::initializeSufficientStatistics(void) {
    std::vector<float> zeros(n_states_*n_features_, 0.0);
    CudaSafeCall(cudaMemcpy(d_obs_,         &zeros[0], n_states_*n_features_*sizeof(float), cudaMemcpyHostToDevice));
    CudaSafeCall(cudaMemcpy(d_obs_squared_, &zeros[0], n_states_*n_features_*sizeof(float), cudaMemcpyHostToDevice));
    CudaSafeCall(cudaMemcpy(d_post_,        &zeros[0], n_states_*sizeof(float), cudaMemcpyHostToDevice));
}

void CUDAGaussianHMM::computeSufficientStatistics() {
    float alpha = 1.0f;
    float beta = 1.0f;
    cublasStatus_t status;

    // Compute the sufficient statistics for the mean, \Sum_i p(X_i in state_k) * X_i 
    // MATRIX_MULTIPLY(posteriors.T, obs)
    status = cublasSgemm(
        (cublasHandle_t) cublas_handle_, CUBLAS_OP_N, CUBLAS_OP_T,
        n_features_, n_states_, n_total_observations_, &alpha,
        d_trajectories_, n_features_,
        d_posteriors_, n_states_,
        &beta, d_obs_, n_features_);
    if (status != CUBLAS_STATUS_SUCCESS) { fprintf(stderr, "cublasSgemm() failed at %s:%i\n", __FILE__, __LINE__); exit(EXIT_FAILURE); }

    // Compute the sufficient statistics for the variance, \Sum_i p(X_i in state_k) * X_i**2 
    // MATRIX_MULTIPLY(posteriors.T, obs**2)
    status = cublasSgemm(
        (cublasHandle_t) cublas_handle_, CUBLAS_OP_N, CUBLAS_OP_T,
        n_features_, n_states_, n_total_observations_, &alpha,
        d_trajectories2_, n_features_,
        d_posteriors_, n_states_,
        &beta, d_obs_squared_, n_features_);
    if (status != CUBLAS_STATUS_SUCCESS) { fprintf(stderr, "cublasSgemm() failed at %s:%i\n", __FILE__, __LINE__); exit(EXIT_FAILURE); }

    // Compute the normalization constant for the posterior weighted averages, \Sum_i (P_X_i in state_k)
    status = cublasSgemm(
        (cublasHandle_t) cublas_handle_, CUBLAS_OP_N, CUBLAS_OP_T,
        1, n_states_, n_total_observations_, &alpha,
        d_ones_, 1,
        d_posteriors_, n_states_,
        &beta, d_post_, 1);
    if (status != CUBLAS_STATUS_SUCCESS) { fprintf(stderr, "cublasSgemm() failed at %s:%i\n", __FILE__, __LINE__); exit(EXIT_FAILURE); }

}

CUDAGaussianHMM::~CUDAGaussianHMM() {
    CudaSafeCall(cudaFree(d_trajectories_));
    CudaSafeCall(cudaFree(d_trajectories2_));
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
    CudaSafeCall(cudaFree(d_ones_));


    CudaSafeCall(cudaFree(d_post_));
    CudaSafeCall(cudaFree(d_obs_));
    CudaSafeCall(cudaFree(d_obs_squared_));
    CudaSafeCall(cudaFree(d_counts_));
    cublasDestroy((cublasHandle_t) cublas_handle_);
}



}  // namespace Mixtape
