#include <Python.h>
#ifndef DL_IMPORT
  #define DL_IMPORT(t) t
#endif
#include "GaussianHMMFitter.h"
#include "gaussian.h"
#include <cmath>

namespace Mixtape {

template <class T>
GaussianHMMFitter<T>::GaussianHMMFitter(void* owner, int n_states, int n_features, int n_iter, const double* log_startprob) :
        HMMFitter<T>(n_states, n_features, n_iter, log_startprob),
        owner(owner), means_over_variances(n_states*n_features), means2_over_variances(n_states*n_features),
        variances(n_states*n_features), log_variances(n_states*n_features) {
}

template <class T>
GaussianHMMFitter<T>::~GaussianHMMFitter() {
}

template <class T>
void GaussianHMMFitter<T>::set_means_and_variances(const double* means, const double* variances) {
    for (int i = 0; i < (int) this->variances.size(); i++) {
        this->variances[i] = variances[i];
        means_over_variances[i] = means[i]/variances[i];
        means2_over_variances[i] = means_over_variances[i]*means[i];
        log_variances[i] = log(variances[i]);
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
                temp += means2_over_variances[j*this->n_features+i]
                        - 2.0*element*means_over_variances[j*this->n_features+i]
                        + element*element/variances[j*this->n_features+i]
                        + log_variances[j*this->n_features+i];
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
    for (int i = 0; i < this->n_states; i++)
        for (int j = 0; j < this->n_features; j++)
            for (int k = 0; k < traj_length; k++) {
                T element = trajectory.get<T>(k, j);
                traj_obs[i*this->n_features+j] += element*posteriors[k][i];
                traj_obs2[i*this->n_features+j] += element*element*posteriors[k][i];
            }
    for (int i = 0; i < this->n_states; i++) {
        for (int j = 0; j < this->n_features; j++) {
            obs[i*this->n_features+j] += traj_obs[i*this->n_features+j];
            obs2[i*this->n_features+j] += traj_obs2[i*this->n_features+j];
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
