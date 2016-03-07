#include <Python.h>
#ifndef DL_IMPORT
  #define DL_IMPORT(t) t
#endif
#define _USE_MATH_DEFINES
#include "GaussianHMMFitter.h"
#include "gaussian.h"
#include <cmath>

namespace msmbuilder {

template <class T>
GaussianHMMFitter<T>::GaussianHMMFitter(void* owner, int n_states, int n_features, int n_iter, const double* log_startprob) :
        HMMFitter<T>(n_states, n_features, n_iter, log_startprob), owner(owner), a0(n_states*n_features), a1(n_states*n_features), a2(n_states*n_features) {
}

template <class T>
GaussianHMMFitter<T>::~GaussianHMMFitter() {
}

template <class T>
void GaussianHMMFitter<T>::set_means_and_variances(const double* means, const double* variances) {
    int n_elements = this->n_states*this->n_features;
    for (int i = 0; i < n_elements; i++) {
        this->a0[i] = means[i]*means[i]/variances[i] + log(variances[i]);
        this->a1[i] = -2.0*means[i]/variances[i];
        this->a2[i] = 1.0/variances[i];
    }
}

template <class T>
void GaussianHMMFitter<T>::initialize_sufficient_statistics() {
    obs.resize(this->n_states*this->n_features);
    obs2.resize(this->n_states*this->n_features);
    for (int i = 0; i < (int) obs.size(); i++) {
        obs[i] = 0;
        obs2[i] = 0;
    }
    this->post.resize(this->n_states);
    for (int i = 0; i < this->n_states; i++)
        this->post[i] = 0;
}

template <class T>
void GaussianHMMFitter<T>::compute_log_likelihood(const Trajectory& trajectory,
                            std::vector<std::vector<double> >& frame_log_probability) const {
    static const float log_M_2_PI = std::log(2*M_PI);
    for (int t = 0; t < trajectory.frames(); t++) {
        for (int j = 0; j < this->n_states; j++) {
            double temp = 0;
            for (int i = 0; i < this->n_features; i++) {
                T element = trajectory.get<T>(t, i);
                int index = j*this->n_features+i;
                temp += a0[index] + element*(a1[index] + element*a2[index]);
            }
            frame_log_probability[t][j] = -0.5*(this->n_features*log_M_2_PI+temp);
        }
    }
}

template <class T>
void GaussianHMMFitter<T>::accumulate_sufficient_statistics(const Trajectory& trajectory,
                                      const std::vector<std::vector<double> >& frame_log_probability,
                                      const std::vector<std::vector<double> >& posteriors,
                                      const std::vector<std::vector<double> >& fwdlattice,
                                      const std::vector<std::vector<double> >& bwdlattice) {
    int traj_length = trajectory.frames();
    std::vector<double> traj_obs(this->n_states*this->n_features, 0);
    std::vector<double> traj_obs2(this->n_states*this->n_features, 0);
    std::vector<double> state_posteriors(traj_length);
    for (int i = 0; i < this->n_states; i++) {
        // Copy the posteriors into a compact array.  This makes memory access more efficient in the inner loop.
        for (int j = 0; j < traj_length; j++)
            state_posteriors[j] = posteriors[j][i];
        for (int j = 0; j < this->n_features; j++) {
            double temp1 = 0.0;
            double temp2 = 0.0;
            for (int k = 0; k < traj_length; k++) {
                T element = trajectory.get<T>(k, j);
                temp1 += element*state_posteriors[k];
                temp2 += element*element*state_posteriors[k];
            }
            traj_obs[i*this->n_features+j] += temp1;
            traj_obs2[i*this->n_features+j] += temp2;
        }
    }
#pragma omp critical
    {
        for (int i = 0; i < this->n_states; i++) {
            for (int j = 0; j < this->n_features; j++) {
                obs[i*this->n_features+j] += traj_obs[i*this->n_features+j];
                obs2[i*this->n_features+j] += traj_obs2[i*this->n_features+j];
            }
        }
    }
}

template <class T>
void GaussianHMMFitter<T>::get_obs(double* output) {
    for (int i = 0; i < this->n_states; i++)
        for (int j = 0; j < this->n_features; j++)
            output[i*this->n_features+j] = obs[i*this->n_features+j];
}

template <class T>
void GaussianHMMFitter<T>::get_obs2(double* output) {
    for (int i = 0; i < this->n_states; i++)
        for (int j = 0; j < this->n_features; j++)
            output[i*this->n_features+j] = obs2[i*this->n_features+j];
}

template <>
void GaussianHMMFitter<float>::do_mstep() {
    GaussianHMMObject* hmm = (GaussianHMMObject*) owner;
    _do_mstep_float(hmm, this);
}

template <>
void GaussianHMMFitter<double>::do_mstep() {
    GaussianHMMObject* hmm = (GaussianHMMObject*) owner;
    _do_mstep_double(hmm, this);
}

template class GaussianHMMFitter<float>;
template class GaussianHMMFitter<double>;

}
