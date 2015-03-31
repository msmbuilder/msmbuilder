#ifndef MIXTAPE_HMM_FITTER_H
#define MIXTAPE_HMM_FITTER_H

#include "Trajectory.h"
#include "logsumexp.hpp"
#include <cmath>
#include <vector>

#include <cstdio>

namespace Mixtape {

template <class T>
class HMMFitter {
public:
    HMMFitter(int n_states, int n_features, int n_iter, const double* log_startprob) :
            n_states(n_states), n_features(n_features), n_iter(n_iter), log_startprob(log_startprob), log_transmat(n_states*n_states),
            transition_counts(n_states, std::vector<double>(n_states, 0)), post(n_states, 0) {
    }

    virtual ~HMMFitter() {
    }
    
    void set_transmat(const double* transmat) {
        for (int i = 0; i < n_states*n_states; i++) {
            log_transmat[i] = std::log(transmat[i]);
        }
    }

    virtual void initialize_sufficient_statistics() = 0;

    virtual void compute_log_likelihood(const Trajectory& trajectory,
                                        std::vector<std::vector<double> >& frame_log_probability) const = 0;

    virtual void accumulate_sufficient_statistics(const Trajectory& trajectory,
                                                  const std::vector<std::vector<double> >& frame_log_probability,
                                                  const std::vector<std::vector<double> >& posteriors,
                                                  const std::vector<std::vector<double> >& fwdlattice,
                                                  const std::vector<std::vector<double> >& bwdlattice) = 0;

    virtual void do_mstep() = 0;
    
    void fit(const std::vector<Trajectory> trajectories, double convergence_threshold) {
        std::vector<std::vector<double> > frame_log_probability, fwdlattice, bwdlattice, posteriors;
        std::vector<double> iter_log_probability;
        for (int i = 0; i < n_iter; i++) {
            // Expectation step
            initialize_sufficient_statistics();
            double current_log_probability = 0.0;
            for (int j = 0; j < (int) trajectories.size(); j++) {
                const Trajectory& trajectory = trajectories[j];
                frame_log_probability.resize(trajectory.frames(), std::vector<double>(n_states));
                fwdlattice.resize(trajectory.frames(), std::vector<double>(n_states));
                bwdlattice.resize(trajectory.frames(), std::vector<double>(n_states));
                posteriors.resize(trajectory.frames(), std::vector<double>(n_states));
                
                compute_log_likelihood(trajectory, frame_log_probability);
                do_forward_pass(frame_log_probability, fwdlattice);
                do_backward_pass(frame_log_probability, bwdlattice);
                compute_posteriors(fwdlattice, bwdlattice, posteriors);
                compute_transition_counts(frame_log_probability, fwdlattice, bwdlattice, transition_counts);
                current_log_probability += logsumexp(&fwdlattice[trajectory.frames()-1][0], n_states);
                for (int state = 0; state < n_states; state++)
                    for (int frame = 0; frame < trajectory.frames(); frame++)
                        post[state] += posteriors[frame][state];
                accumulate_sufficient_statistics(trajectory, frame_log_probability, posteriors, fwdlattice, bwdlattice);
            }
            iter_log_probability.push_back(current_log_probability);

            // Check for convergence
            if (i > 0 && fabs(iter_log_probability[i]-iter_log_probability[i-1]) < convergence_threshold)
                break;

            // Maximization step
            do_mstep();
        }
    }
    
    void get_transition_counts(double* output) {
        for (int i = 0; i < n_states; i++)
            for (int j = 0; j < n_states; j++)
                output[i*n_states+j] = transition_counts[i][j];
    }
    
    void get_post(double* output) {
        for (int i = 0; i < this->n_states; i++)
            output[i] = post[i];
    }
protected:
    int n_states, n_features, n_iter;
    const double* log_startprob;
    std::vector<double> log_transmat;
    std::vector<std::vector<double> > transition_counts;
    std::vector<double> post;
    
    void do_forward_pass(const std::vector<std::vector<double> >& frame_log_probability,
                         std::vector<std::vector<double> >& fwdlattice) {
        for (int i = 0; i < n_states; i++)
            fwdlattice[0][i] = log_startprob[i] + frame_log_probability[0][i];
        std::vector<double> work_buffer(n_states);
        for (int t = 1; t < (int) fwdlattice.size(); t++) {
            for (int j = 0; j < n_states; j++) {
                for (int i = 0; i < n_states; i++)
                    work_buffer[i] = fwdlattice[t-1][i] + log_transmat[i*n_states+j];
                fwdlattice[t][j] = logsumexp(&work_buffer[0], n_states) + frame_log_probability[t][j];
            }
        }
    }
    
    void do_backward_pass(const std::vector<std::vector<double> >& frame_log_probability,
                          std::vector<std::vector<double> >& bwdlattice) {
        int sequence_length = bwdlattice.size();
        for (int i = 0; i < n_states; i++)
            bwdlattice[sequence_length-1][i] = 0;
        std::vector<double> work_buffer(n_states);
        for (int t = sequence_length-2; t >= 0; t--) {
            for (int i = 0; i < n_states; i++) {
                for (int j = 0; j < n_states; j++)
                    work_buffer[j] = frame_log_probability[t+1][j] + bwdlattice[t+1][j] + log_transmat[i*n_states+j];
                bwdlattice[t][i] = logsumexp(&work_buffer[0], n_states);
            }
        }
    }
    
    void compute_posteriors(const std::vector<std::vector<double> >& fwdlattice,
                            const std::vector<std::vector<double> >& bwdlattice,
                            std::vector<std::vector<double> >& posteriors) {
        std::vector<double> gamma(n_states);
        for (int t = 0; t < (int) fwdlattice.size(); t++) {
            for (int i = 0; i < n_states; i++)
                gamma[i] = fwdlattice[t][i] + bwdlattice[t][i];
            double normalizer = logsumexp(&gamma[0], n_states);
            for (int i = 0; i < n_states; i++)
                posteriors[t][i] = std::exp(gamma[i]-normalizer);
        }
    }
    
    void compute_transition_counts(const std::vector<std::vector<double> >& frame_log_probability,
                                     const std::vector<std::vector<double> >& fwdlattice,
                                     const std::vector<std::vector<double> >& bwdlattice,
                                     std::vector<std::vector<double> >& transition_counts) {
        int sequence_length = fwdlattice.size();
        std::vector<double> work_buffer(sequence_length);
        double logprob = logsumexp(&fwdlattice[sequence_length-1][0], n_states);
        for (int i = 0; i < n_states; i++) {
            for (int j = 0; j < n_states; j++) {
                for (int t = 0; t < sequence_length-1; t++)
                    work_buffer[t] = fwdlattice[t][i] + log_transmat[i*n_states+j] + frame_log_probability[t+1][j] + bwdlattice[t+1][j] - logprob;
                transition_counts[i][j] = std::exp(logsumexp(&work_buffer[0], sequence_length-1));
            }
        }
    }
};

} // namespace Mixtape

#endif