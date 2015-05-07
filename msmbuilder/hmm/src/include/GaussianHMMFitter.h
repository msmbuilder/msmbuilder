#ifndef MIXTAPE_GAUSSIAN_HMM_FITTER_H
#define MIXTAPE_GAUSSIAN_HMM_FITTER_H

#include "HMMFitter.h"

namespace msmbuilder {

/**
 * This subclass of HMMFitter computes Gaussian HMMs.
 */
template <class T>
class GaussianHMMFitter : public HMMFitter<T> {
public:
    GaussianHMMFitter(void* owner, int n_states, int n_features, int n_iter, const double* log_startprob);
    
    ~GaussianHMMFitter();
    
    void set_means_and_variances(const double* means, const double* variances);
    
    void initialize_sufficient_statistics();
    
    void compute_log_likelihood(const Trajectory& trajectory,
                                std::vector<std::vector<double> >& frame_log_probability) const;

    void accumulate_sufficient_statistics(const Trajectory& trajectory,
                                          const std::vector<std::vector<double> >& frame_log_probability,
                                          const std::vector<std::vector<double> >& posteriors,
                                          const std::vector<std::vector<double> >& fwdlattice,
                                          const std::vector<std::vector<double> >& bwdlattice);
    
    void get_obs(double* output);
    
    void get_obs2(double* output);

    void do_mstep();
private:
    void* owner;
    std::vector<double> obs, obs2, a0, a1, a2;
};

} // namespace msmbuilder

#endif