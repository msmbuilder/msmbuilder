#include <stdlib.h>
#include <stdio.h>
#include <vector>
#include <string>
#include "CUDAGaussianHMM.hpp"

#include <cublas_v2.h>
#include "forward.cu"
#include "backward.cu"
#include "gaussian_likelihood.cu"
#include "posteriors.cu"
#include "expectedtransitions.cu"


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

void cudaMalloc2(void** devicePtr, size_t nbytes) {
    CudaSafeCall(cudaMalloc(devicePtr, nbytes));
    CudaSafeCall(cudaMemset(*devicePtr, 0x55, nbytes));
}

CUDAGaussianHMM::CUDAGaussianHMM(
    const float** sequences,
    const int n_sequences,
    const int* sequence_lengths,
    const int n_states,
    const int n_features)
    : sequences_(sequences)
    , n_sequences_(n_sequences)
    , n_observations_(0)
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
    , d_sequence_lengths_(NULL)
    , d_cum_sequence_lengths_(NULL)
{
    sequence_lengths_.resize(n_sequences);
    cum_sequence_lengths_.resize(n_sequences);
    for (int i = 0; i < n_sequences_; i++) {
        sequence_lengths_[i] = sequence_lengths[i];
        n_observations_ += sequence_lengths[i];
        if (i == 0)
            cum_sequence_lengths_[i] = 0;
        else
            cum_sequence_lengths_[i] = cum_sequence_lengths_[i-1] + sequence_lengths[i-1];
    }
    
    // Arrays of size proportional to the number of observations
    cudaMalloc2((void **) &d_sequences_, n_observations_*n_features_*sizeof(float));
    cudaMalloc2((void **) &d_sequences2_, n_observations_*n_features_*sizeof(float));
    cudaMalloc2((void **) &d_fwdlattice_, n_observations_*n_states_*sizeof(float));
    cudaMalloc2((void **) &d_bwdlattice_, n_observations_*n_states_*sizeof(float));
    cudaMalloc2((void **) &d_posteriors_, n_observations_*n_states_*sizeof(float));
    cudaMalloc2((void **) &d_framelogprob_, n_observations_*n_states_*sizeof(float));
    cudaMalloc2((void **) &d_ones_, n_observations_*sizeof(float));

    // Small data arrays
    cudaMalloc2((void **) &d_log_transmat_, n_states_*n_states_*sizeof(float));
    cudaMalloc2((void **) &d_log_transmat_T_, n_states_*n_states_*sizeof(float));
    cudaMalloc2((void **) &d_means_, n_states_*n_features_*sizeof(float));
    cudaMalloc2((void **) &d_variances_, n_states_*n_features_*sizeof(float));
    cudaMalloc2((void **) &d_log_startprob_, n_states_*sizeof(float));
    cudaMalloc2((void **) &d_sequence_lengths_, n_sequences_*sizeof(float));
    cudaMalloc2((void **) &d_cum_sequence_lengths_, n_sequences_*sizeof(float));

    // Sufficient statistics
    cudaMalloc2((void **) &d_post_, n_states_*sizeof(float));
    cudaMalloc2((void **) &d_obs_, n_states_*n_features_*sizeof(float));
    cudaMalloc2((void **) &d_obs_squared_, n_states_*n_features_*sizeof(float));
    cudaMalloc2((void **) &d_transcounts_, n_states_*n_states_*sizeof(float));

    // Sequence data
    for (int i = 0; i < n_sequences_; i++) {
        int n = sequence_lengths_[i]*n_features_;
        int offset = cum_sequence_lengths_[i]*n_features_;
        CudaSafeCall(cudaMemcpy(d_sequences_ + offset, sequences_[i], n*sizeof(float), cudaMemcpyHostToDevice));
    }

    CudaSafeCall(cudaMemcpy(d_sequence_lengths_, &sequence_lengths_[0],
                            n_sequences_*sizeof(float), cudaMemcpyHostToDevice));
    CudaSafeCall(cudaMemcpy(d_cum_sequence_lengths_, &cum_sequence_lengths_[0],
                            n_sequences_*sizeof(float), cudaMemcpyHostToDevice));

    cublasStatus_t status = cublasCreate((cublasHandle_t*) &cublas_handle_);
    if (status != CUBLAS_STATUS_SUCCESS) { exit(EXIT_FAILURE); }
    
    fill<<<1, 256>>>(d_ones_, 1.0, n_observations_);
    square<<<1, 256>>>(d_sequences_, n_observations_*n_features_,
                       d_sequences2_);
    cudaDeviceSynchronize();
    CudaCheckError();
}


float CUDAGaussianHMM::computeEStep() {
    gaussian_likelihood<<<1, 32>>>(
        d_sequences_, d_means_, d_variances_, n_sequences_,
        d_sequence_lengths_, d_cum_sequence_lengths_, n_states_,
        n_features_, d_framelogprob_);

    forward4<<<1, 32>>>(
        d_log_transmat_T_, d_log_startprob_, d_framelogprob_,
        d_sequence_lengths_, d_cum_sequence_lengths_, n_sequences_,
        d_fwdlattice_);
    backward4<<<1, 32>>>(
        d_log_transmat_T_, d_log_startprob_, d_framelogprob_,
        d_sequence_lengths_, d_cum_sequence_lengths_, n_sequences_,
        d_bwdlattice_);
    cudaDeviceSynchronize();
    posteriors4<<<1, 32>>>(
        d_fwdlattice_, d_bwdlattice_, n_sequences_,
        d_sequence_lengths_, d_cum_sequence_lengths_, d_posteriors_);

    cudaDeviceSynchronize();
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
    CudaSafeCall(cudaMemcpy(d_log_startprob_, &log_startprob[0],
                 n_states_*sizeof(float), cudaMemcpyHostToDevice));
}

void CUDAGaussianHMM::getFrameLogProb(float* out) {
    CudaSafeCall(cudaMemcpy(out, d_framelogprob_, n_observations_*n_states_*sizeof(float), cudaMemcpyDeviceToHost));
}

void CUDAGaussianHMM::getFwdLattice(float* out) {
    CudaSafeCall(cudaMemcpy(out, d_fwdlattice_, n_observations_*n_states_*sizeof(float), cudaMemcpyDeviceToHost));
}

void CUDAGaussianHMM::getBwdLattice(float* out) {
    CudaSafeCall(cudaMemcpy(out, d_bwdlattice_, n_observations_*n_states_*sizeof(float), cudaMemcpyDeviceToHost));
}

void CUDAGaussianHMM::getPosteriors(float* out) {
    CudaSafeCall(cudaMemcpy(out, d_posteriors_, n_observations_*n_states_*sizeof(float), cudaMemcpyDeviceToHost));
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

void CUDAGaussianHMM::getStatsTransCounts(float* out) {
    CudaSafeCall(cudaMemcpy(out, d_transcounts_, n_states_*n_states_*sizeof(float), cudaMemcpyDeviceToHost));
}

void CUDAGaussianHMM::initializeSufficientStatistics(void) {
    CudaSafeCall(cudaMemset(d_obs_, 0, n_states_*n_features_*sizeof(float)));
    CudaSafeCall(cudaMemset(d_obs_squared_, 0, n_states_*n_features_*sizeof(float)));
    CudaSafeCall(cudaMemset(d_post_, 0, n_states_*sizeof(float)));
    CudaSafeCall(cudaMemset(d_transcounts_, 0, n_states_*n_states_*sizeof(float)));
}

void CUDAGaussianHMM::computeSufficientStatistics() {
    float alpha = 1.0f;
    float beta = 1.0f;
    cublasStatus_t status;
    
    // Compute the sufficient statistics for the mean,
    // \Sum_i p(X_i in state_k) * X_i 
    // MATRIX_MULTIPLY(posteriors.T, obs)
    status = cublasSgemm(
        (cublasHandle_t) cublas_handle_, CUBLAS_OP_N, CUBLAS_OP_T,
        n_features_, n_states_, n_observations_, &alpha,
        d_sequences_, n_features_,
        d_posteriors_, n_states_,
        &beta, d_obs_, n_features_);

    if (status != CUBLAS_STATUS_SUCCESS) { fprintf(stderr, "cublasSgemm() failed at %s:%i\n", __FILE__, __LINE__); exit(EXIT_FAILURE); }

    // Compute the sufficient statistics for the variance,
    // \Sum_i p(X_i in state_k) * X_i**2 
    // MATRIX_MULTIPLY(posteriors.T, obs**2)
    status = cublasSgemm(
        (cublasHandle_t) cublas_handle_, CUBLAS_OP_N, CUBLAS_OP_T,
        n_features_, n_states_, n_observations_, &alpha,
        d_sequences2_, n_features_,
        d_posteriors_, n_states_,
        &beta, d_obs_squared_, n_features_);
    if (status != CUBLAS_STATUS_SUCCESS) { fprintf(stderr, "cublasSgemm() failed at %s:%i\n", __FILE__, __LINE__); exit(EXIT_FAILURE); }

    // Compute the normalization constant for the posterior weighted 
    // averages, \Sum_i (P_X_i in state_k)
    status = cublasSgemm(
        (cublasHandle_t) cublas_handle_, CUBLAS_OP_N, CUBLAS_OP_T,
        1, n_states_, n_observations_, &alpha,
        d_ones_, 1,
        d_posteriors_, n_states_,
        &beta, d_post_, 1);
    if (status != CUBLAS_STATUS_SUCCESS) { fprintf(stderr, "cublasSgemm() failed at %s:%i\n", __FILE__, __LINE__); exit(EXIT_FAILURE); }

    transitioncounts<<<1, 32>>>(
        d_fwdlattice_, d_bwdlattice_, d_log_transmat_, d_framelogprob_,
        n_observations_, n_states_, d_transcounts_);
    
    cudaDeviceSynchronize();
    CudaCheckError();
}

CUDAGaussianHMM::~CUDAGaussianHMM() {
    CudaSafeCall(cudaFree(d_fwdlattice_));
    CudaSafeCall(cudaFree(d_bwdlattice_));
    CudaSafeCall(cudaFree(d_posteriors_));
    CudaSafeCall(cudaFree(d_framelogprob_));
    CudaSafeCall(cudaFree(d_log_transmat_));
    CudaSafeCall(cudaFree(d_log_transmat_T_));
    CudaSafeCall(cudaFree(d_means_));
    CudaSafeCall(cudaFree(d_variances_));
    CudaSafeCall(cudaFree(d_log_startprob_));
    CudaSafeCall(cudaFree(d_sequence_lengths_));
    CudaSafeCall(cudaFree(d_cum_sequence_lengths_));
    CudaSafeCall(cudaFree(d_ones_));
    CudaSafeCall(cudaFree(d_post_));
    CudaSafeCall(cudaFree(d_obs_));
    CudaSafeCall(cudaFree(d_obs_squared_));
    CudaSafeCall(cudaFree(d_transcounts_));
    CudaSafeCall(cudaFree(d_sequences_));
    CudaSafeCall(cudaFree(d_sequences2_));

    cublasDestroy((cublasHandle_t) cublas_handle_);
}



}  // namespace Mixtape
