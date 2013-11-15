"""
A simple implementation of the Kalman Filter, Kalman Smoother, and EM
algorithm for Linear-Gaussian state space models.

Primarily adapted from Daniel Duckworth's pykalman library.
"""
import warnings

import numpy as np
import numpy.random
from numpy import shape, zeros
from scipy import linalg

# Simple Utility functions
def array1d(X, dtype=None, order=None):
    """Returns at least 1-d array with data from X"""
    return np.asarray(np.atleast_1d(X), dtype=dtype, order=order)


def array2d(X, dtype=None, order=None):
    """Returns at least 2-d array with data from X"""
    return np.asarray(np.atleast_2d(X), dtype=dtype, order=order)

def _determine_dimensionality(variables, default):
    """Derive the dimensionality of the state space

    Parameters
    ----------
    variables : list of ({None, array}, conversion function, index)
        variables, functions to convert them to arrays, and indices in those
        arrays to derive dimensionality from.
    default : {None, int}
        default dimensionality to return if variables is empty

    Returns
    -------
    dim : int
        dimensionality of state space as derived from variables or default.
    """
    # gather possible values based on the variables
    candidates = []
    for (v, converter, idx) in variables:
        if v is not None:
            v = converter(v)
            candidates.append(v.shape[idx])

    # also use the manually specified default
    if default is not None:
        candidates.append(default)

    # ensure consistency of all derived values
    if len(candidates) == 0:
        return 1
    else:
        if not np.all(np.array(candidates) == candidates[0]):
            raise ValueError(
                "The shape of all " +
                "parameters is not consistent.  " +
                "Please re-check their values."
            )
        return candidates[0]

class KalmanFilter(object):
  """ Implements Kalman Filter, Kalman Smoother, and EM algorithm for
      linear Gaussian models
  """
  def __init__(self, transition_matrix=None, observation_matrix=None,
      transition_covariance=None, observation_covariance=None,
      transition_offset=None, observation_offset=None,
      initial_state_mean=None, initial_state_covariance=None,
      em_vars=['transition_matrix', 'transition_covariance',
        'observation_matrix', 'observation_covariance',
        'initial_state_mean', 'initial_state_covariance'],
      n_dim_state=None, n_dim_obs=None):
    n_dim_state = _determine_dimensionality(
        [(transition_matrix, array2d, -2),
         (transition_offset, array1d, -1),
         (transition_covariance, array2d, -2),
         (initial_state_mean, array1d, -1),
         (initial_state_covariance, array2d, -2),
         (observation_matrix, array2d, -1)],
        n_dim_state
    )
    n_dim_obs = _determine_dimensionality(
        [(observation_matrix, array2d, -2),
         (observation_offset, array1d, -1),
         (observation_covariance, array2d, -2)],
        n_dim_obs
    )
    # Save the input matrices
    self.transition_matrix = transition_matrix
    self.observation_matrix = observation_matrix
    self.transition_covariance = transition_covariance
    self.observation_covariance = observation_covariance
    self.transition_offset = transition_offset
    self.observation_offset = observation_offset
    self.initial_state_mean = initial_state_mean
    self.initial_state_covariance = initial_state_covariance
    self.em_vars = em_vars
    self.n_dim_state = n_dim_state
    self.n_dim_obs = n_dim_obs

  def sample(self, n_timesteps, initial_state=None):
    """ Sample a state sequence"""
    transition_matrix = self.transition_matrix
    transition_offset = self.transition_offset
    transition_covariance = self.transition_covariance
    observation_matrix = self.observation_matrix
    observation_offset = self.observation_offset
    observation_covariance = self.observation_covariance
    initial_state_mean = self.initial_state_mean
    initial_state_covariance = self.initial_state_covariance

    n_dim_state = self.n_dim_state
    n_dim_obs = self.n_dim_obs
    states = zeros((n_timesteps, n_dim_state))
    observations = zeros((n_timesteps, n_dim_obs))

    # Sample initial state
    if initial_state is None:
      initial_state = numpy.random.multivariate_normal(
          initial_state_mean, initial_state_covariance)

    # Generate the samples
    for t in range(n_timesteps):
      if t == 0:
        states[t] = initial_state
      else:
        states[t] = np.dot(transition_matrix, states[t - 1]) + \
             transition_offset + \
             numpy.random.multivariate_normal(
                np.zeros(n_dim_state),
                transition_covariance)

        observations[t] = np.dot(observation_matrix, states[t]) +\
            observation_offset + \
            numpy.random.multivariate_normal( np.zeros(n_dim_obs),
                observation_covariance)

    return states, observations

  def filter(self, observations):
    """Perform the Kalman filter
    Parameters
    __________
    observations : observations corresponding to times [0...n_timesteps-1]
    Returns
    _______
    filtered_state_means
    filtered_state_covariances
    """
    transition_matrix = self.transition_matrix
    transition_offset = self.transition_offset
    transition_covariance = self.transition_covariance
    observation_matrix = self.observation_matrix
    observation_offset = self.observation_offset
    observation_covariance = self.observation_covariance
    initial_state_mean = self.initial_state_mean
    initial_state_covariance = self.initial_state_covariance

    n_timesteps = observations.shape[0]
    n_dim_state = self.n_dim_state
    n_dim_obs = self.n_dim_obs

    predicted_state_means = zeros((n_timesteps, n_dim_state))
    predicted_state_covariances = zeros((n_timesteps, n_dim_state,
                                    n_dim_state))
    kalman_gains = zeros((n_timesteps, n_dim_state, n_dim_obs))
    filtered_state_means = zeros((n_timesteps, n_dim_state))
    filtered_state_covariances = zeros((n_timesteps, n_dim_state,
                                  n_dim_state))
    for t in range(n_timesteps):
      if t == 0:
        predicted_state_means[t] = initial_state_mean
        predicted_state_covariances[t] = initial_state_covariance
      else:
        predicted_state_means[t], predicted_state_covariances[t] = \
            self._filter_predict(transition_matrix, transition_covariance,
                transition_offset, filtered_state_means[t-1],
                filtered_state_covariances[t-1])
        (kalman_gains[t], filtered_state_means[t],
            filtered_state_covariances[t]) = self._filter_correct(
                observation_matrix, observation_covariance,
                observation_offset, predicted_state_means[t],
                predicted_state_covariances[t], observations[t])
    return (predicted_state_means, predicted_state_covariances,
        kalman_gains, filtered_state_means, filtered_state_covariances)

  def _filter_predict(self, transition_matrix, transition_covariance,
        transition_offset, current_state_mean, current_state_covariance):
    """Perform the forward prediction step of the kalman filter."""
    predicted_state_mean = np.dot(transition_matrix, current_state_mean) +\
        transition_offset
    predicted_state_covariance = np.dot(transition_matrix,
        np.dot(current_state_covariance, transition_matrix.T)) +\
            transition_covariance
    return (predicted_state_mean, predicted_state_covariance)


  def _filter_correct(self, observation_matrix, observation_covariance,
      observation_offset, predicted_state_mean, predicted_state_covariance,
      observation):
    """Perform the correctino for the current evidence"""
    predicted_observation_mean = np.dot(observation_matrix,
        predicted_state_mean) + observation_offset
    predicted_observation_covariance = np.dot(observation_matrix,
        np.dot(predicted_state_covariance, observation_matrix.T)) +\
        observation_covariance
    kalman_gain = np.dot(predicted_state_covariance,
        np.dot(observation_matrix.T,
          linalg.pinv(predicted_observation_covariance)))
    corrected_state_mean = predicted_state_mean + \
        np.dot(kalman_gain, observation - predicted_observation_mean)
    corrected_state_covariance = predicted_state_covariance -\
        np.dot(kalman_gain, np.dot(observation_matrix,
          predicted_state_covariance))
    return (kalman_gain, corrected_state_mean, corrected_state_covariance)

  def smooth(self, observations):
    """Apply the Kalman Smoother"""
    transition_matrix = self.transition_matrix
    transition_offset = self.transition_offset
    transition_covariance = self.transition_covariance
    observation_matrix = self.observation_matrix
    observation_offset = self.observation_offset
    observation_covariance = self.observation_covariance
    initial_state_mean = self.initial_state_mean
    initial_state_covariance = self.initial_state_covariance

    (predicted_state_means, predicted_state_covariances, _,
        filtered_state_means, filtered_state_covariances) = \
            self.filter(observations)
    n_timesteps, n_dim_state = shape(filtered_state_means)

    smoothed_state_means = zeros((n_timesteps, n_dim_state))
    smoothed_state_covariances = zeros((n_timesteps, n_dim_state,
      n_dim_state))
    kalman_smoothing_gains = zeros((n_timesteps-1, n_dim_state,
      n_dim_state))

    smoothed_state_means[-1] = filtered_state_means[-1]
    smoothed_state_covariances[-1] = filtered_state_covariances[-1]

    for t in reversed(range(n_timesteps-1)):
      (smoothed_state_means[t], smoothed_state_covariances[t],
          kalman_smoothing_gains[t]) = self._smooth_update(
              transition_matrix, filtered_state_means[t],
              filtered_state_covariances[t], predicted_state_means[t+1],
              predicted_state_covariances[t+1], smoothed_state_means[t+1],
              smoothed_state_covariances[t+1])
    return (smoothed_state_means, smoothed_state_covariances,
        kalman_smoothing_gains)

  def _smooth_update(self, transition_matrix, filtered_state_mean,
      filtered_state_covariance, predicted_state_mean,
      predicted_state_covariance, next_smoothed_state_mean,
      next_smoothed_state_covariance):
    """Perform the backwards smoothing update"""
    kalman_smoothing_gain = np.dot(filtered_state_covariance,
        np.dot(transition_matrix.T,
          linalg.pinv(predicted_state_covariance)))
    smoothed_state_mean = filtered_state_mean +\
        np.dot(kalman_smoothing_gain,
            next_smoothed_state_mean - predicted_state_mean)
    smoothed_state_covariance = filtered_state_covariance + np.dot(
        kalman_smoothing_gain, np.dot(
          next_smoothed_state_covariance - predicted_state_covariance,
          kalman_smoothing_gain.T))
    return (smoothed_state_mean, smoothed_state_covariance,
      kalman_smoothing_gain)
