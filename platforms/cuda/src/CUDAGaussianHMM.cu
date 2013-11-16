#include <stdlib.h>
#include <stdio.h>
#include "utils.h"
#include "CUDAGaussianHMM.hpp"
#include "forward.cuh"
namespace Mixtape {

float CUDAGaussianHMM::doEStep() {
    //    gaussian_loglikelihood_diag(
    //trajectories_, &means_[0], &variances_[0],
    //n_trajectories_, &n_observations_[0],
    //n_states_, n_features_, &framelogprob_[0]);
    do_forward<<<1, 32>>>(
        d_log_transmat_T_, d_log_startprob_, d_framelogprob_,
        n_trajectories_, d_trj_offset_, d_n_observations_,
        n_states_, d_fwdlattice_);
    /*
    do_backward(
        &log_transmat_[0], &log_startprob_[0], &framelogprob_[0],
        n_trajectories_, &n_observations_[0], n_states_, &bwdlattice_[0]);
    do_posteriors(
        &fwdlattice_[0], &bwdlattice_[0], n_trajectories_,
        &n_observations_[0], n_states_, &posteriors_[0]);

    int ptr = 0;
    float logprob = 0;
    for (int s = 0; s < n_trajectories_; s++) {
        logprob += logsumexp(&fwdlattice_[ptr + (n_observations_[s]-1)*n_states_], n_states_);
        ptr += n_observations_[s]*n_states_;
    }*/

    //    return logprob;
    return 1.0;

}

}  // namespace Mixtape
