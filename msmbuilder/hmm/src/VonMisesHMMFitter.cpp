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




#include <cstdlib>



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
                            std::vector<std::vector<double> >& frame_log_probability) const {
  double *kappa_cos_means, *kappa_sin_means;
  double val, log_numerator, cos_obs_kj, sin_obs_kj;
  const double LOG_2PI = 1.8378770664093453;
  int n_states = this->n_states;
  int n_features = this->n_features;

  // clear the output
  for (int i = 0; i < trajectory.frames(); i++)
      for (int j = 0; j < n_states; j++)
          frame_log_probability[i][j] = 0.0;
  // allocate two workspaces
  kappa_cos_means = (double*) malloc(n_states * n_features * sizeof(double));
  kappa_sin_means = (double*) malloc(n_states * n_features * sizeof(double));
  if (NULL == kappa_cos_means || NULL == kappa_sin_means) {
    fprintf(stderr, "compute_log_likelihood: Memory allocation failure");
    exit(EXIT_FAILURE);
  }

  // this sets the likelihood's denominator
  for (int i = 0; i < n_states; i++) {
    for (int j = 0; j < n_features; j++) {
      val = LOG_2PI + log(i0(kappas[i*n_features + j]));
      for (int k = 0; k < trajectory.frames(); k++)
        frame_log_probability[k][i] -= val;
    }
  }

  // We need to calculate cos(obs[k*n_features + j] - means[i*n_features + j])
  // But we want to avoid having a trig function in the inner tripple loop,
  // so we use the double angle formula to split up the computation into cos(x)cos(y) + sin(x)*sin(y)
  // where each of the terms can be computed in a double loop.
  for (int i = 0; i < n_states; i++) {
    for (int j = 0; j < n_features; j++) {
      kappa_cos_means[j*n_states + i] = kappas[i*n_features + j] * cos(means[i*n_features + j]);
      kappa_sin_means[j*n_states + i] = kappas[i*n_features + j] * sin(means[i*n_features + j]);
    }
  }

  for (int k = 0; k < trajectory.frames(); k++) {
    for (int j = 0; j < n_features; j++) {
      T element = trajectory.get<T>(k, j);
      cos_obs_kj = cos(element);
      sin_obs_kj = sin(element);
      for (int i = 0; i < n_states; i++) {
        log_numerator = (cos_obs_kj*kappa_cos_means[j*n_states + i] + 
        sin_obs_kj*kappa_sin_means[j*n_states + i]);
        frame_log_probability[k][i] += log_numerator;
      }
    }
  }

  free(kappa_cos_means);
  free(kappa_sin_means);
}

template <class T>
void VonMisesHMMFitter<T>::accumulate_sufficient_statistics(const Trajectory& trajectory,
                                      const std::vector<std::vector<double> >& frame_log_probability,
                                      const std::vector<std::vector<double> >& posteriors,
                                      const std::vector<std::vector<double> >& fwdlattice,
                                      const std::vector<std::vector<double> >& bwdlattice) {
    int traj_length = trajectory.frames();
    std::vector<double> traj_cosobs(this->n_states*this->n_features, 0);
    std::vector<double> traj_sinobs(this->n_states*this->n_features, 0);
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
                temp1 += cos(element)*state_posteriors[k];
                temp2 += sin(element)*state_posteriors[k];
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
