#include <stdlib.h>
#include <stdio.h>

#include "CPUGaussianHMM.hpp"

#include "logsumexp.h"
#include "gaussian_likelihood.h"
#include "forward.h"
#include "backward.h"
#include "posteriors.h"
namespace Mixtape {

float CPUGaussianHMM::doMStep() {
    if ((means_.size() != (n_states_ * n_features_)) ||
        (variances_.size() != (n_states_ * n_features_)) ||
        (log_transmat_.size() != (n_states_ * n_states_)) ||
        (log_transmat_T_.size() != (n_states_ * n_states_)) ||
        (log_startprob_.size() != (n_states_))) {
        fprintf(stderr, "Uninitiailized\n");
        exit(EXIT_FAILURE);
    }

    gaussian_loglikelihood_diag(
        trajectories_, &means_[0], &variances_[0],
        n_trajectories_, &n_observations_[0],
        n_states_, n_features_, &framelogprob_[0]);
    do_forward(
        &log_transmat_T_[0], &log_startprob_[0], &framelogprob_[0],
        n_trajectories_, &n_observations_[0], n_states_, &fwdlattice_[0]);
    do_backward(
        &log_transmat_[0], &log_startprob_[0], &framelogprob_[0],
        n_trajectories_, &n_observations_[0], n_states_, &bwdlattice_[0]);
    do_posteriors(
        &fwdlattice_[0], &bwdlattice_[0], n_trajectories_,
        &n_observations_[0], n_states_, &posteriors_[0]);

    int ptr = 0;
    float logprob = 0;
    for (int s = 0; s < n_trajectories_; s++) {
        logprob += logsumexp(&fwdlattice_[ptr + (n_observations_[s]-2)*n_states_], n_states_);
        ptr += n_observations_[s]*n_states_;
    }

    return logprob;
        
}

}  // namespace Mixtape
