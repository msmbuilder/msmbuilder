#include <Python.h>
#ifndef DL_IMPORT
  #define DL_IMPORT(t) t
#endif
#define _USE_MATH_DEFINES
#include "VonMisesHMMFitter.h"
#include "vonmises.h"
#include <cmath>
extern "C" {
#include "cephes.h"
}

using namespace std;

namespace msmbuilder {

template <class T>
VonMisesHMMFitter<T>::VonMisesHMMFitter(void* owner, int n_states, int n_features, int n_iter, const double* log_startprob) :
        HMMFitter<T>(n_states, n_features, n_iter, log_startprob), owner(owner), means(n_states*n_features), kappas(n_states*n_features) {
}

template <class T>
VonMisesHMMFitter<T>::~VonMisesHMMFitter() {
}

template <class T>
void VonMisesHMMFitter<T>::set_means_and_kappas(const double* means, const double* kappas) {
    int n_elements = this->n_states*this->n_features;
    for (int i = 0; i < n_elements; i++) {
        this->means[i] = means[i];
        this->kappas[i] = kappas[i];
    }
}

template <class T>
void VonMisesHMMFitter<T>::initialize_sufficient_statistics() {
    cosobs.resize(this->n_states*this->n_features);
    sinobs.resize(this->n_states*this->n_features);
    for (int i = 0; i < (int) cosobs.size(); i++) {
        cosobs[i] = 0;
        sinobs[i] = 0;
    }
    this->post.resize(this->n_states);
    for (int i = 0; i < this->n_states; i++)
        this->post[i] = 0;
}

template <class T>
void VonMisesHMMFitter<T>::compute_log_likelihood(const Trajectory& trajectory,
                            vector<vector<double> >& frame_log_probability) const {
    const double LOG_2PI = log(2*M_PI);
    int n_states = this->n_states;
    int n_features = this->n_features;
    int traj_length = trajectory.frames();

    // Clear the output
    for (int i = 0; i < traj_length; i++)
        for (int j = 0; j < n_states; j++)
            frame_log_probability[i][j] = 0.0;

    // Transpose the log probability matrix, since this makes memory access more efficient in the next loop.
    vector<double> frame_log_probability_T(n_states*traj_length);
    for (int i = 0; i < n_states; i++)
        for (int j = 0; j < traj_length; j++)
            frame_log_probability_T[i*traj_length+j] = frame_log_probability[j][i];

    // This sets the likelihood's denominator
    for (int i = 0; i < n_states; i++) {
        for (int j = 0; j < n_features; j++) {
            double val = LOG_2PI + log(i0(kappas[i*n_features + j]));
            for (int k = 0; k < traj_length; k++)
                frame_log_probability_T[i*traj_length+k] -= val;
        }
    }
    for (int i = 0; i < n_states; i++)
        for (int j = 0; j < traj_length; j++)
            frame_log_probability[j][i] = frame_log_probability_T[i*traj_length+j];

    // We need to calculate cos(obs[k*n_features + j] - means[i*n_features + j])
    // But we want to avoid having a trig function in the inner triple loop,
    // so we use the double angle formula to split up the computation into cos(x)cos(y) + sin(x)*sin(y)
    // where each of the terms can be computed in a double loop.
    vector<double> kappa_cos_means(n_states*n_features);
    vector<double> kappa_sin_means(n_states*n_features);
    for (int i = 0; i < n_states; i++) {
        for (int j = 0; j < n_features; j++) {
            kappa_cos_means[j*n_states + i] = kappas[i*n_features + j] * cos(means[i*n_features + j]);
            kappa_sin_means[j*n_states + i] = kappas[i*n_features + j] * sin(means[i*n_features + j]);
        }
    }

    for (int k = 0; k < traj_length; k++) {
        for (int j = 0; j < n_features; j++) {
            T element = trajectory.get<T>(k, j);
            double cos_obs_kj = cos(element);
            double sin_obs_kj = sin(element);
            for (int i = 0; i < n_states; i++) {
                double log_numerator = (cos_obs_kj*kappa_cos_means[j*n_states + i] + sin_obs_kj*kappa_sin_means[j*n_states + i]);
                frame_log_probability[k][i] += log_numerator;
            }
        }
    }
}

template <class T>
void VonMisesHMMFitter<T>::accumulate_sufficient_statistics(const Trajectory& trajectory,
                                      const vector<vector<double> >& frame_log_probability,
                                      const vector<vector<double> >& posteriors,
                                      const vector<vector<double> >& fwdlattice,
                                      const vector<vector<double> >& bwdlattice) {
    int traj_length = trajectory.frames();
    vector<double> traj_cosobs(this->n_states*this->n_features, 0);
    vector<double> traj_sinobs(this->n_states*this->n_features, 0);
    vector<double> coselement(traj_length*this->n_features);
    vector<double> sinelement(traj_length*this->n_features);
    vector<double> state_posteriors(traj_length);
    
    // Precompute the sin and cosine of every element of the trajectory.
    
    for (int i = 0; i < this->n_features; i++)
        for (int j = 0; j < traj_length; j++) {
            T element = trajectory.get<T>(j, i);
            coselement[i*traj_length+j] = cos(element);
            sinelement[i*traj_length+j] = sin(element);
        }
    
    // Main loop to accumulate statistics.
    
    for (int i = 0; i < this->n_states; i++) {
        // Copy the posteriors into a compact array.  This makes memory access more efficient in the inner loop.
        for (int j = 0; j < traj_length; j++)
            state_posteriors[j] = posteriors[j][i];
        for (int j = 0; j < this->n_features; j++) {
            double temp1 = 0.0;
            double temp2 = 0.0;
            for (int k = 0; k < traj_length; k++) {
                temp1 += coselement[j*traj_length+k]*state_posteriors[k];
                temp2 += sinelement[j*traj_length+k]*state_posteriors[k];
            }
            traj_cosobs[i*this->n_features+j] += temp1;
            traj_sinobs[i*this->n_features+j] += temp2;
        }
    }
#pragma omp critical
    {
        for (int i = 0; i < this->n_states; i++)
            for (int j = 0; j < this->n_features; j++) {
                cosobs[i*this->n_features+j] += traj_cosobs[i*this->n_features+j];
                sinobs[i*this->n_features+j] += traj_sinobs[i*this->n_features+j];
            }
    }
}

template <class T>
void VonMisesHMMFitter<T>::get_cosobs(double* output) {
    for (int i = 0; i < this->n_states; i++)
        for (int j = 0; j < this->n_features; j++)
            output[i*this->n_features+j] = cosobs[i*this->n_features+j];
}

template <class T>
void VonMisesHMMFitter<T>::get_sinobs(double* output) {
    for (int i = 0; i < this->n_states; i++)
        for (int j = 0; j < this->n_features; j++)
            output[i*this->n_features+j] = sinobs[i*this->n_features+j];
}

template <>
void VonMisesHMMFitter<float>::do_mstep() {
    VonMisesHMMObject* hmm = (VonMisesHMMObject*) owner;
    _do_mstep_float(hmm, this);
}

template <>
void VonMisesHMMFitter<double>::do_mstep() {
    VonMisesHMMObject* hmm = (VonMisesHMMObject*) owner;
    _do_mstep_double(hmm, this);
}

template class VonMisesHMMFitter<float>;
template class VonMisesHMMFitter<double>;

}
