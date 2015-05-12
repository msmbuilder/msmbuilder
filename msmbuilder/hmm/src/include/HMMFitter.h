#ifndef MIXTAPE_HMM_FITTER_H
#define MIXTAPE_HMM_FITTER_H

#include "Trajectory.h"
#include "logsumexp.hpp"
#include <algorithm>
#include <cmath>
#include <vector>

namespace msmbuilder {

/**
 * This class fits Hidden Markov Models to trajectories.  It is an abstract class that defines the features common to
 * all types of HMMs.  Subclasses specialize it for particular types of HMMs.
 *
 * The template argument is the type of data contained in the trajectories.
 */
template <class T>
class HMMFitter {
public:
    /**
     * Create a new HMMFitter.
     *
     * @param n_states       the number of states in the model
     * @param n_features     the number of features in the model
     * @param n_iter         the maximum number of iterations of the EM algorithm to perform
     * @param log_startprob  the log of the initial probability for each state
     */
    HMMFitter(int n_states, int n_features, int n_iter, const double* log_startprob) :
            n_states(n_states), n_features(n_features), n_iter(n_iter), log_startprob(log_startprob), log_transmat(n_states*n_states),
            transition_counts(n_states, std::vector<double>(n_states, 0)), post(n_states, 0) {
    }

    virtual ~HMMFitter() {
    }
    
    /** Set the transition matrix. */
    void set_transmat(const double* transmat) {
        for (int i = 0; i < n_states*n_states; i++) {
            log_transmat[i] = std::log(std::max(transmat[i], 1e-20));
        }
    }

    /**
     * Initialize the sufficient statistics are the start of an iteration.  Subclasses must implement this.
     */
    virtual void initialize_sufficient_statistics() = 0;

    /**
     * Compute the log likelihood of each state in each frame of a trajectory.  Subclasses must implement this.
     */
    virtual void compute_log_likelihood(const Trajectory& trajectory,
                                        std::vector<std::vector<double> >& frame_log_probability) const = 0;

    /**
     * Accumulate the sufficient statistics computed during an iteration.  Subclasses must implement this.
     */
    virtual void accumulate_sufficient_statistics(const Trajectory& trajectory,
                                                  const std::vector<std::vector<double> >& frame_log_probability,
                                                  const std::vector<std::vector<double> >& posteriors,
                                                  const std::vector<std::vector<double> >& fwdlattice,
                                                  const std::vector<std::vector<double> >& bwdlattice) = 0;

    /**
     * Perform the M step of an iteration.  Subclasses must implement this.
     */
    virtual void do_mstep() = 0;
    
    /**
     * Fit the model to a set of Trajectories.
     *
     * @param trajectories           the set of Trajectories to fit it to
     * @param convergence_threshold  iteration will stop when the log probability changes by less than this amount
     */
    void fit(const std::vector<Trajectory>& trajectories, double convergence_threshold) {
        iter_log_probability.clear();
        for (int i = 0; i < n_iter; i++) {
            // Expectation step
            initialize_sufficient_statistics();
            double current_log_probability = 0.0;
            for (int i = 0; i < n_states; i++)
                for (int j = 0; j < n_states; j++)
                    transition_counts[i][j] = 0.0;
#pragma omp parallel for default(shared)
            for (int j = 0; j < (int) trajectories.size(); j++) {
                const Trajectory& trajectory = trajectories[j];
                std::vector<std::vector<double> > frame_log_probability(trajectory.frames(), std::vector<double>(n_states));
                std::vector<std::vector<double> > fwdlattice(trajectory.frames(), std::vector<double>(n_states));
                std::vector<std::vector<double> > bwdlattice(trajectory.frames(), std::vector<double>(n_states));
                std::vector<std::vector<double> > posteriors(trajectory.frames(), std::vector<double>(n_states));
                std::vector<std::vector<double> > traj_transition_counts(n_states, std::vector<double>(n_states));
                compute_log_likelihood(trajectory, frame_log_probability);
                do_forward_pass(frame_log_probability, fwdlattice);
                do_backward_pass(frame_log_probability, bwdlattice);
                compute_posteriors(fwdlattice, bwdlattice, posteriors);
                compute_transition_counts(frame_log_probability, fwdlattice, bwdlattice, traj_transition_counts);
                double traj_log_probability = logsumexp(&fwdlattice[trajectory.frames()-1][0], n_states);
                accumulate_sufficient_statistics(trajectory, frame_log_probability, posteriors, fwdlattice, bwdlattice);
#pragma omp critical
                {
                    for (int k = 0; k < n_states; k++) {
                        for (int m = 0; m < n_states; m++)
                            transition_counts[k][m] += traj_transition_counts[k][m];
                        for (int frame = 0; frame < trajectory.frames(); frame++)
                            post[k] += posteriors[frame][k];
                    }
                    current_log_probability += traj_log_probability;
                }
            }
            iter_log_probability.push_back(current_log_probability);

            // Check for convergence
            if (i > 0 && fabs(iter_log_probability[i]-iter_log_probability[i-1]) < convergence_threshold)
                break;

            // Maximization step
            do_mstep();
        }
    }
    
    /**
     * Compute the log probability of a set of Trajectories given this model.
     * 
     * @param trajectories           the set of Trajectories to score
     */
    double score_trajectories(const std::vector<Trajectory>& trajectories) const {
        std::vector<std::vector<double> > frame_log_probability, fwdlattice;
        double log_probability = 0.0;
        for (int j = 0; j < (int) trajectories.size(); j++) {
            const Trajectory& trajectory = trajectories[j];
            frame_log_probability.resize(trajectory.frames(), std::vector<double>(n_states));
            fwdlattice.resize(trajectory.frames(), std::vector<double>(n_states));
            compute_log_likelihood(trajectory, frame_log_probability);
            do_forward_pass(frame_log_probability, fwdlattice);
            log_probability += logsumexp(&fwdlattice[trajectory.frames()-1][0], n_states);
        }
        return log_probability;
    }
    
    /**
     * Use the Viterbi algorithm to compute the most likely sequence of states to have produced a given Trajectory.
     * 
     * @param trajectory        the Trajectory for which to predict the sequence of states
     * @param state_sequence    the state indices are stored into this array
     * @returns the log probability of the predicted state sequence according to the model
     */
    double predict_state_sequence(const Trajectory& trajectory, int* state_sequence) const {
        std::vector<std::vector<double> > frame_log_probability(trajectory.frames(), std::vector<double>(n_states));
        compute_log_likelihood(trajectory, frame_log_probability);
        std::vector<std::vector<double> > viterbi_lattice(trajectory.frames(), std::vector<double>(n_states));
        std::vector<std::vector<double> > work_buffer(n_states, std::vector<double>(n_states));
        
        // Initialization.
        
        for (int i = 0; i < n_states; i++)
            viterbi_lattice[0][i] = log_startprob[i]+frame_log_probability[0][i];
        
        // Induction.
        
        for (int t = 1; t < trajectory.frames(); t++) {
            for (int i = 0; i < n_states; i++)
                for (int j = 0; j < n_states; j++)
                    work_buffer[i][j] = viterbi_lattice[t-1][j]+log_transmat[j*n_states+i]; // ???
            for (int i = 0; i < n_states; i++)
                viterbi_lattice[t][i] = *std::max_element(work_buffer[i].begin(), work_buffer[i].end()) + frame_log_probability[t][i];
        }
        
        // Observation traceback.
        
        int max_pos = 0;
        for (int i = 1; i < n_states; i++)
            if (viterbi_lattice.back()[i] > viterbi_lattice.back()[max_pos])
                max_pos = i;
        state_sequence[trajectory.frames()-1] = max_pos;
        double logprob = viterbi_lattice.back()[max_pos];
        for (int t = trajectory.frames()-2; t >= 0; t--) {
            max_pos = 0;
            for (int i = 1; i < n_states; i++) {
                double value = viterbi_lattice[t][i]+log_transmat[i*n_states+state_sequence[t+1]];
                if (viterbi_lattice[t][i]+log_transmat[i*n_states+state_sequence[t+1]] > viterbi_lattice[t][max_pos]+log_transmat[max_pos*n_states+state_sequence[t+1]])
                    max_pos = i;
            }
            state_sequence[t] = max_pos;
        }
        return logprob;
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
    
    void get_log_probability(double* output) {
        for (int i = 0; i < (int) iter_log_probability.size(); i++)
            output[i] = iter_log_probability[i];
    }
    
    int get_fit_iterations() {
        return iter_log_probability.size();
    }
protected:
    int n_states, n_features, n_iter;
    const double* log_startprob;
    std::vector<double> log_transmat, iter_log_probability;
    std::vector<std::vector<double> > transition_counts;
    std::vector<double> post;
    
    void do_forward_pass(const std::vector<std::vector<double> >& frame_log_probability,
                         std::vector<std::vector<double> >& fwdlattice) const {
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
                          std::vector<std::vector<double> >& bwdlattice) const {
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
                            std::vector<std::vector<double> >& posteriors) const {
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
                                     std::vector<std::vector<double> >& transition_counts) const {
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

} // namespace msmbuilder

#endif
