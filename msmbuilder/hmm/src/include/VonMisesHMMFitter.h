#ifndef MIXTAPE_GAUSSIAN_HMM_FITTER_H
#define MIXTAPE_GAUSSIAN_HMM_FITTER_H

#include "HMMFitter.h"

namespace msmbuilder {

/**
 * This subclass of HMMFitter computes von Mises HMMs.
 */
template <class T>
class VonMisesHMMFitter : public HMMFitter<T> {
public:
    VonMisesHMMFitter(void* owner, int n_states, int n_features, int n_iter, const double* log_startprob);
    
    ~VonMisesHMMFitter();
    
    void set_means_and_kappas(const double* means, const double* kappas);
    
    void initialize_sufficient_statistics();
    
    void compute_log_likelihood(const Trajectory& trajectory,
                                std::vector<std::vector<double> >& frame_log_probability) const;

    void accumulate_sufficient_statistics(const Trajectory& trajectory,
                                          const std::vector<std::vector<double> >& frame_log_probability,
                                          const std::vector<std::vector<double> >& posteriors,
                                          const std::vector<std::vector<double> >& fwdlattice,
                                          const std::vector<std::vector<double> >& bwdlattice);
    
    void get_cosobs(double* output);

    void get_sinobs(double* output);
    
    void do_mstep();
private:
    void* owner;
    std::vector<double> cosobs, sinobs, means, kappas;
};

} // namespace msmbuilder

#endif