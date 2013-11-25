/*****************************************************************/
/*    Copyright (c) 2013, Stanford University and the Authors    */
/*    Author: Robert McGibbon <rmcgibbo@gmail.com>               */
/*    Contributors:                                              */
/*                                                               */
/*****************************************************************/

#include <cstdlib>
#include <cmath>
#include <cstdio>
#include <vector>
#include <limits>
#include <string>
#define CUDA_ERROR_CHECK
#include "safecuda.hpp"
#include "MixtapeException.hpp"
#include "CUDAGaussianHMM.hpp"

#include <cublas_v2.h>
#include "forward.cu"
#include "backward.cu"
#include "gaussian_likelihood.cu"
#include "posteriors.cu"
#include "expectedtransitions.cu"
#include "sufficientstatistics.cu"

//#define USE_CUBLAS
#define NEW_MVN_KERNEL


namespace Mixtape {
CUDAGaussianHMM::CUDAGaussianHMM(const int n_states,
                                 const int n_features)
    : n_sequences_(0)
    , n_states_(n_states)
    // we have special kernels for n_states in [4, 8, 16, 32], and then a
    // generic one for more than 32 states, so if the number of states is less,
    // need to "pad" to the next power in the range.
    , n_pstates_(n_states < 32 ? max(4, static_cast<int>(pow(2, ceil(log(n_states)/log(2))))) : n_states)
    , n_observations_(0)
    , n_features_(n_features)
    , d_fwdlattice_(NULL)
    , d_bwdlattice_(NULL)
    , d_posteriors_(NULL)
    , d_framelogprob_(NULL)
    , d_log_transmat_(NULL)
    , d_log_transmat_T_(NULL)
    , d_means_(NULL)
    , d_variances_(NULL)
    , d_logvariances_(NULL)
    , d_log_startprob_(NULL)
    , d_logprob_(NULL)
    , d_sequence_lengths_(NULL)
    , d_cum_sequence_lengths_(NULL)
    , d_ones_(NULL)
    , d_post_(NULL)
    , d_obs_(NULL)
    , d_obs_squared_(NULL)
    , d_transcounts_(NULL)
    , d_sequences_(NULL)
    , d_sequences2_(NULL)
{
    // Small data arrays
    cudaMalloc2((void **) &d_log_transmat_, n_pstates_*n_pstates_*sizeof(float));
    cudaMalloc2((void **) &d_log_transmat_T_, n_pstates_*n_pstates_*sizeof(float));
    cudaMalloc2((void **) &d_means_, n_pstates_*n_features_*sizeof(float));
    cudaMalloc2((void **) &d_variances_, n_pstates_*n_features_*sizeof(float));
    cudaMalloc2((void **) &d_logvariances_, n_pstates_*n_features_*sizeof(float));
    cudaMalloc2((void **) &d_log_startprob_, n_pstates_*sizeof(float));
    cudaMalloc2((void **) &d_logprob_, sizeof(float));

    // Sufficient statistics
    cudaMalloc2((void **) &d_post_, n_pstates_*sizeof(float));
    cudaMalloc2((void **) &d_obs_, n_pstates_*n_features_*sizeof(float));
    cudaMalloc2((void **) &d_obs_squared_, n_pstates_*n_features_*sizeof(float));
    cudaMalloc2((void **) &d_transcounts_, n_pstates_*n_pstates_*sizeof(float));

    cublasStatus_t status = cublasCreate((cublasHandle_t*) &cublas_handle_);
    if (status != CUBLAS_STATUS_SUCCESS) { throw MixtapeException("cuBLAS initialization error."); }
    cudaDeviceSynchronize();
}

void CUDAGaussianHMM::setSequences(const float** sequences,
                                   const int n_sequences,
                                   const int* sequence_lengths)
{
    n_observations_ = 0;
    n_sequences_ = n_sequences;
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
    delSequences();

    // Arrays of size proportional to the number of observations
    cudaMalloc2((void **) &d_sequences_, n_observations_*n_features_*sizeof(float));
#ifdef USE_CUBLAS
    cudaMalloc2((void **) &d_sequences2_, n_observations_*n_features_*sizeof(float));
#endif
    cudaMalloc2((void **) &d_fwdlattice_, n_observations_*n_pstates_*sizeof(float));
    cudaMalloc2((void **) &d_bwdlattice_, n_observations_*n_pstates_*sizeof(float));
    cudaMalloc2((void **) &d_posteriors_, n_observations_*n_pstates_*sizeof(float));
    cudaMalloc2((void **) &d_framelogprob_, n_observations_*n_pstates_*sizeof(float));
    cudaMalloc2((void **) &d_ones_, n_observations_*sizeof(float));

    cudaMalloc2((void **) &d_sequence_lengths_, n_sequences_*sizeof(float));
    cudaMalloc2((void **) &d_cum_sequence_lengths_, n_sequences_*sizeof(float));

    // Copy over sequence data
    for (int i = 0; i < n_sequences_; i++) {
        int n = sequence_lengths_[i]*n_features_;
        int offset = cum_sequence_lengths_[i]*n_features_;
        CudaSafeCall(cudaMemcpy(d_sequences_ + offset, sequences[i], n*sizeof(float), cudaMemcpyHostToDevice));
    }

    CudaSafeCall(cudaMemcpy(d_sequence_lengths_, &sequence_lengths_[0],
                            n_sequences_*sizeof(float), cudaMemcpyHostToDevice));
    CudaSafeCall(cudaMemcpy(d_cum_sequence_lengths_, &cum_sequence_lengths_[0],
                            n_sequences_*sizeof(float), cudaMemcpyHostToDevice));
    fill<<<100, 256>>>(d_ones_, 1.0, n_observations_);
#ifdef USE_CUBLAS
    square<<<1, 256>>>(d_sequences_, n_observations_*n_features_, d_sequences2_);
#endif
    cudaDeviceSynchronize();
}

void CUDAGaussianHMM::delSequences() {
    if (d_sequences_ != NULL)
        cudaFree(d_sequences_);
#ifdef USE_CUBLAS
    if (d_sequences2_ != NULL)
        cudaFree(d_sequences2_);
#endif
    if (d_fwdlattice_ != NULL)
        cudaFree(d_fwdlattice_);
    if (d_bwdlattice_ != NULL)
        cudaFree(d_bwdlattice_);
    if (d_posteriors_ != NULL)
        cudaFree(d_posteriors_);
    if (d_framelogprob_ != NULL)
        cudaFree(d_framelogprob_);
    if (d_ones_ != NULL)
        cudaFree(d_ones_);
    if (d_sequence_lengths_ != NULL)
        cudaFree(d_sequence_lengths_);
    if (d_cum_sequence_lengths_ != NULL)
        cudaFree(d_cum_sequence_lengths_);
}

float CUDAGaussianHMM::computeEStep() {
    if (d_sequences_ == NULL)
        throw MixtapeException("Sequence data not initialized");

#ifdef NEW_MVN_KERNEL
    fill<<<100, 256>>>(d_framelogprob_, 0.0, n_observations_*n_pstates_);
#else
    gaussian_likelihood<<<n_sequences_, 256>>>(
         d_sequences_, d_means_, d_variances_, n_sequences_,
         d_sequence_lengths_, d_cum_sequence_lengths_, n_pstates_,
         n_features_, d_framelogprob_);
#endif
    cudaDeviceSynchronize();


    if (n_pstates_ == 4) {
#ifdef NEW_MVN_KERNEL
        log_diag_mvn_likelihood<4,16><<<256, 64>>>(
            d_sequences_, d_means_,  d_variances_, d_logvariances_,
            n_observations_, n_pstates_, n_features_, d_framelogprob_);
        cudaDeviceSynchronize();
#endif
        forward4<<<max(1, n_sequences_/8), 256>>>(
            d_log_transmat_T_, d_log_startprob_, d_framelogprob_,
            d_sequence_lengths_, d_cum_sequence_lengths_, n_sequences_,
            d_fwdlattice_);
        backward4<<<max(1, n_sequences_/8), 256>>>(
            d_log_transmat_, d_log_startprob_, d_framelogprob_,
            d_sequence_lengths_, d_cum_sequence_lengths_, n_sequences_,
            d_bwdlattice_);
        cudaDeviceSynchronize();

        posteriors<4><<<max(1, n_sequences_/8), 256>>>(
            d_fwdlattice_, d_bwdlattice_, n_sequences_,
            d_sequence_lengths_, d_cum_sequence_lengths_, d_posteriors_);
    }
    else if (n_pstates_ == 8) {
#ifdef NEW_MVN_KERNEL
        log_diag_mvn_likelihood<8,16><<<256, 128>>>(
            d_sequences_, d_means_,  d_variances_, d_logvariances_,
            n_observations_, n_pstates_, n_features_, d_framelogprob_);
        cudaDeviceSynchronize();
#endif

        forward8<<<max(1, n_sequences_/8) , 256>>>(
            d_log_transmat_T_, d_log_startprob_, d_framelogprob_,
            d_sequence_lengths_, d_cum_sequence_lengths_, n_sequences_,
            d_fwdlattice_);
        backward8<<<max(1, n_sequences_/8), 256>>>(
            d_log_transmat_, d_log_startprob_, d_framelogprob_,
            d_sequence_lengths_, d_cum_sequence_lengths_, n_sequences_,
            d_bwdlattice_);
        cudaDeviceSynchronize();

        posteriors<8><<<max(1, n_sequences_/8), 256>>>(
            d_fwdlattice_, d_bwdlattice_, n_sequences_,
            d_sequence_lengths_, d_cum_sequence_lengths_, d_posteriors_);
    } else if (n_pstates_ == 16) {
#ifdef NEW_MVN_KERNEL
        log_diag_mvn_likelihood<8,16><<<256, 128>>>(
            d_sequences_, d_means_,  d_variances_, d_logvariances_,
            n_observations_, n_pstates_, n_features_, d_framelogprob_);
        cudaDeviceSynchronize();
#endif

        forward16<<<max(1, n_sequences_/8), 256>>>(
            d_log_transmat_T_, d_log_startprob_, d_framelogprob_,
            d_sequence_lengths_, d_cum_sequence_lengths_, n_sequences_,
            d_fwdlattice_);
        backward16<<<max(1, n_sequences_/8), 256>>>(
            d_log_transmat_, d_log_startprob_, d_framelogprob_,
            d_sequence_lengths_, d_cum_sequence_lengths_, n_sequences_,
            d_bwdlattice_);

        cudaDeviceSynchronize();
        posteriors<16><<<max(1, n_sequences_/8), 256>>>(
            d_fwdlattice_, d_bwdlattice_, n_sequences_,
            d_sequence_lengths_, d_cum_sequence_lengths_, d_posteriors_);
    } else if (n_pstates_ == 32) {
#ifdef NEW_MVN_KERNEL
        log_diag_mvn_likelihood<8,16><<<256, 128>>>(
            d_sequences_, d_means_,  d_variances_, d_logvariances_,
            n_observations_, n_pstates_, n_features_, d_framelogprob_);
        cudaDeviceSynchronize();
#endif
        forward32<<<max(1, n_sequences_/8), 256>>>(
            d_log_transmat_T_, d_log_startprob_, d_framelogprob_,
            d_sequence_lengths_, d_cum_sequence_lengths_, n_sequences_,
            d_fwdlattice_);
        backward32<<<max(1, n_sequences_/8), 256>>>(
            d_log_transmat_, d_log_startprob_, d_framelogprob_,
            d_sequence_lengths_, d_cum_sequence_lengths_, n_sequences_,
            d_bwdlattice_);

        cudaDeviceSynchronize();
        posteriors<32><<<max(1, n_sequences_/8), 256>>>(
            d_fwdlattice_, d_bwdlattice_, n_sequences_,
            d_sequence_lengths_, d_cum_sequence_lengths_, d_posteriors_);

    } else {
        throw MixtapeException("n_states > 32 is not implemented yet");
    }

    cudaDeviceSynchronize();
    CudaCheckError();
    return 1.0;
}

void CUDAGaussianHMM::setMeans(const float* means) {
    if (n_pstates_ != n_states_)
        // we need to fill some values in for the means corresponding to the
        // padding states. Otherwise, nans propagate through the system
        fill<<<1, 256>>>(d_means_ + n_states_*n_features_,
                         std::numeric_limits<float>::max(),
                         (n_pstates_-n_states_) * n_features_);
    CudaSafeCall(cudaMemcpy(d_means_, means, n_states_*n_features_*sizeof(float), cudaMemcpyHostToDevice));
    cudaDeviceSynchronize();
}

void CUDAGaussianHMM::getMeans(float* out) {
    CudaSafeCall(cudaMemcpy(out, d_means_, n_states_*n_features_*sizeof(float), cudaMemcpyDeviceToHost));
}

void CUDAGaussianHMM::setVariances(const float* variances) {
    std::vector<float> vars(n_pstates_*n_features_, -1.0);
    std::vector<float> logvars(n_pstates_*n_features_, 1.0);

    for (int i = 0; i < n_states_*n_features_; i++) {
        vars[i] = variances[i];
        logvars[i] = log(variances[i]);
    }
    CudaSafeCall(cudaMemcpy(d_variances_, &vars[0], n_pstates_*n_features_*sizeof(float), cudaMemcpyHostToDevice));
    CudaSafeCall(cudaMemcpy(d_logvariances_, &logvars[0], n_pstates_*n_features_*sizeof(float), cudaMemcpyHostToDevice));

    cudaDeviceSynchronize();
}

void CUDAGaussianHMM::getVariances(float* out) {
    CudaSafeCall(cudaMemcpy(out, d_variances_, n_states_*n_features_*sizeof(float), cudaMemcpyDeviceToHost));
}

void CUDAGaussianHMM::setTransmat(const float* transmat) {
    std::vector<float> log_transmat(n_pstates_*n_pstates_);
    std::vector<float> log_transmat_T(n_pstates_*n_pstates_);
    for (int i = 0; i < n_states_; i++)
        for (int j = 0; j < n_states_; j++)
            log_transmat[i*n_pstates_ + j] = log(transmat[i*n_states_ + j]);

    for (int i = n_states_; i < n_pstates_; i++)
        for (int j = n_states_; j < n_pstates_; j++)
            log_transmat[i*n_pstates_ + j] = -std::numeric_limits<float>::max();

    for (int i = 0; i < n_pstates_; i++)
        for (int j = 0; j < n_pstates_; j++)
            log_transmat_T[j*n_pstates_ +i] = log_transmat[i*n_pstates_ + j];

    CudaSafeCall(cudaMemcpy(d_log_transmat_, &log_transmat[0],     n_pstates_*n_pstates_*sizeof(float), cudaMemcpyHostToDevice));
    CudaSafeCall(cudaMemcpy(d_log_transmat_T_, &log_transmat_T[0], n_pstates_*n_pstates_*sizeof(float), cudaMemcpyHostToDevice));
}

void CUDAGaussianHMM::getTransmat(float* out) {
    std::vector<float> log_transmat(n_pstates_*n_pstates_);
    CudaSafeCall(cudaMemcpy(&log_transmat[0], d_log_transmat_, n_pstates_*n_pstates_*sizeof(float), cudaMemcpyDeviceToHost));
    for (int i = 0; i < n_states_; i++)
        for (int j = 0; j < n_states_; j++)
            out[i*n_states_ + j] = exp(log_transmat[i*n_pstates_ + j]);
}

void CUDAGaussianHMM::setStartProb(const float* startProb) {
    std::vector<float> log_startprob(n_pstates_);
    for (int i = 0; i < n_states_; i++)
        log_startprob[i] = log(startProb[i]);
    for (int i = n_states_; i < n_pstates_; i++)
        log_startprob[i] = -std::numeric_limits<float>::max();

    CudaSafeCall(cudaMemcpy(d_log_startprob_, &log_startprob[0],
                 n_pstates_*sizeof(float), cudaMemcpyHostToDevice));
}

void CUDAGaussianHMM::getStartProb(float* out) {
    std::vector<float> log_startprob(n_states_);
    CudaSafeCall(cudaMemcpy(&log_startprob[0], d_log_startprob_, n_states_*sizeof(float), cudaMemcpyDeviceToHost));
    for (int i = 0; i < n_states_; i++)
        out[i] = exp(log_startprob[i]);
}

void CUDAGaussianHMM::getFrameLogProb(float* out) {
    std::vector<float> buf(n_observations_*n_pstates_);
    CudaSafeCall(cudaMemcpy(&buf[0], d_framelogprob_, n_observations_*n_pstates_*sizeof(float), cudaMemcpyDeviceToHost));
    for (int i = 0; i < n_observations_; i++)
        for (int j = 0; j < n_states_; j++)
            out[i*n_states_ + j] = buf[i*n_pstates_ + j];
}

void CUDAGaussianHMM::getFwdLattice(float* out) {
    std::vector<float> buf(n_observations_*n_pstates_);
    CudaSafeCall(cudaMemcpy(&buf[0], d_fwdlattice_, n_observations_*n_pstates_*sizeof(float), cudaMemcpyDeviceToHost));
    for (int i = 0; i < n_observations_; i++)
        for (int j = 0; j < n_states_; j++)
            out[i*n_states_ + j] = buf[i*n_pstates_ + j];
}

void CUDAGaussianHMM::getBwdLattice(float* out) {
    std::vector<float> buf(n_observations_*n_pstates_);
    CudaSafeCall(cudaMemcpy(&buf[0], d_bwdlattice_, n_observations_*n_pstates_*sizeof(float), cudaMemcpyDeviceToHost));
    for (int i = 0; i < n_observations_; i++)
        for (int j = 0; j < n_states_; j++)
            out[i*n_states_ + j] = buf[i*n_pstates_ + j];
}

void CUDAGaussianHMM::getPosteriors(float* out) {
    std::vector<float> buf(n_observations_*n_pstates_);
    CudaSafeCall(cudaMemcpy(&buf[0], d_posteriors_, n_observations_*n_pstates_*sizeof(float), cudaMemcpyDeviceToHost));
    for (int i = 0; i < n_observations_; i++)
        for (int j = 0; j < n_states_; j++)
            out[i*n_states_ + j] = buf[i*n_pstates_ + j];
}

void CUDAGaussianHMM::getStatsObs(float* out) {
    // we just avoid fetching the padding states, since all of the valid
    // memory is contiguous in 1d, since n_states is the major axis
    CudaSafeCall(cudaMemcpy(out, d_obs_, n_states_*n_features_*sizeof(float), cudaMemcpyDeviceToHost));
}

void CUDAGaussianHMM::getStatsObsSquared(float* out) {
    // we just avoid fetching the padding states, since all of the valid
    // memory is contiguous in 1d, since n_states is the major axis
    CudaSafeCall(cudaMemcpy(out, d_obs_squared_, n_states_*n_features_*sizeof(float), cudaMemcpyDeviceToHost));
}

void CUDAGaussianHMM::getStatsPost(float* out) {
    // we just avoid fetching the padding states, since all of the valid
    // memory is contiguous in 1d
    CudaSafeCall(cudaMemcpy(out, d_post_, n_states_*sizeof(float), cudaMemcpyDeviceToHost));
}

void CUDAGaussianHMM::getStatsTransCounts(float* out) {
    std::vector<float> transcounts(n_pstates_*n_pstates_);
    CudaSafeCall(cudaMemcpy(&transcounts[0], d_transcounts_, n_pstates_*n_pstates_*sizeof(float), cudaMemcpyDeviceToHost));
    for (int i = 0; i < n_states_; i++)
        for (int j = 0; j < n_states_; j++)
            out[i*n_states_ + j] = transcounts[i*n_pstates_ + j];
}

void CUDAGaussianHMM::initializeSufficientStatistics(void) {
    CudaSafeCall(cudaMemset(d_obs_, 0, n_pstates_*n_features_*sizeof(float)));
    CudaSafeCall(cudaMemset(d_obs_squared_, 0, n_pstates_*n_features_*sizeof(float)));
    CudaSafeCall(cudaMemset(d_post_, 0, n_pstates_*sizeof(float)));
    CudaSafeCall(cudaMemset(d_transcounts_, 0, n_pstates_*n_pstates_*sizeof(float)));
    CudaSafeCall(cudaMemset(d_logprob_, 0, sizeof(float)));
}

float CUDAGaussianHMM::computeSufficientStatistics() {
#ifdef USE_CUBLAS
    float alpha = 1.0f;
    float beta = 1.0f;
    cublasStatus_t status;

    // Compute the sufficient statistics for the mean,
    // \Sum_i p(X_i in state_k) * X_i
    // MATRIX_MULTIPLY(posteriors.T, obs)
    status = cublasSgemm(
        (cublasHandle_t) cublas_handle_, CUBLAS_OP_N, CUBLAS_OP_T,
        n_features_, n_pstates_, n_observations_, &alpha,
        d_sequences_, n_features_,
        d_posteriors_, n_pstates_,
        &beta, d_obs_, n_features_);
    if (status != CUBLAS_STATUS_SUCCESS) { fprintf(stderr, "cublasSgemm() failed at %s:%i code=%s\n", __FILE__, __LINE__, _cudaGetErrorEnum(status)); exit(EXIT_FAILURE); }

    // Compute the sufficient statistics for the variance,
    // \Sum_i p(X_i in state_k) * X_i**2
    // MATRIX_MULTIPLY(posteriors.T, obs**2)
    status = cublasSgemm(
        (cublasHandle_t) cublas_handle_, CUBLAS_OP_N, CUBLAS_OP_T,
        n_features_, n_pstates_, n_observations_, &alpha,
        d_sequences2_, n_features_,
        d_posteriors_, n_pstates_,
        &beta, d_obs_squared_, n_features_);
    if (status != CUBLAS_STATUS_SUCCESS) { fprintf(stderr, "cublasSgemm() failed at %s:%i\n", __FILE__, __LINE__); exit(EXIT_FAILURE); }

    // Compute the normalization constant for the posterior weighted
    // averages, \Sum_i (P_X_i in state_k)
    status = cublasSgemm(
        (cublasHandle_t) cublas_handle_, CUBLAS_OP_N, CUBLAS_OP_T,
        1, n_pstates_, n_observations_, &alpha,
        d_ones_, 1,
        d_posteriors_, n_pstates_,
        &beta, d_post_, 1);
    if (status != CUBLAS_STATUS_SUCCESS) { fprintf(stderr, "cublasSgemm() failed at %s:%i\n", __FILE__, __LINE__); exit(EXIT_FAILURE); }
#else
    sufficientstatistics<4, 128><<<100, 128>>>(
        d_posteriors_, d_sequences_, n_observations_, n_pstates_, n_features_,
        d_obs_, d_obs_squared_, d_post_);
#endif

    switch (n_pstates_) {
    case 4:
        transitioncounts4_8_16<4, 256><<<max(1, n_sequences_/16), 256>>>(
            d_fwdlattice_, d_bwdlattice_, d_log_transmat_, d_framelogprob_,
            d_sequence_lengths_, d_cum_sequence_lengths_, n_sequences_,
            d_transcounts_, d_logprob_);
        break;
    case 8:
        transitioncounts4_8_16<8, 256><<<max(1, n_sequences_/16), 256>>>(
            d_fwdlattice_, d_bwdlattice_, d_log_transmat_, d_framelogprob_,
            d_sequence_lengths_, d_cum_sequence_lengths_, n_sequences_,
            d_transcounts_, d_logprob_);
        break;
    case 16:
        transitioncounts4_8_16<16, 256><<<max(1, n_sequences_/16), 256>>>(
            d_fwdlattice_, d_bwdlattice_, d_log_transmat_, d_framelogprob_,
            d_sequence_lengths_, d_cum_sequence_lengths_, n_sequences_,
            d_transcounts_, d_logprob_);
        break;
    default:
        throw MixtapeException("NotImplementedError");
    }

    float logprob = 0;
    CudaSafeCall(cudaMemcpy(&logprob, d_logprob_, sizeof(float), cudaMemcpyDeviceToHost));
    cudaDeviceSynchronize();
    CudaCheckError();
    return logprob;
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
    CudaSafeCall(cudaFree(d_logvariances_));
    CudaSafeCall(cudaFree(d_log_startprob_));
    CudaSafeCall(cudaFree(d_logprob_));
    CudaSafeCall(cudaFree(d_sequence_lengths_));
    CudaSafeCall(cudaFree(d_cum_sequence_lengths_));
    CudaSafeCall(cudaFree(d_ones_));
    CudaSafeCall(cudaFree(d_post_));
    CudaSafeCall(cudaFree(d_obs_));
    CudaSafeCall(cudaFree(d_obs_squared_));
    CudaSafeCall(cudaFree(d_transcounts_));
    CudaSafeCall(cudaFree(d_sequences_));
#ifdef USE_CUBLAS
    CudaSafeCall(cudaFree(d_sequences2_));
#endif

    cublasDestroy((cublasHandle_t) cublas_handle_);
}



}  // namespace Mixtape
