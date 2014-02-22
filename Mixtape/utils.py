import json
import numpy as np
from numpy import *
from numpy.linalg import norm
from numpy.random import randint
import mdtraj as md

def iterobjects(fn):
    for line in open(fn, 'r'):
        if line.startswith('#'):
            continue
        try:
            yield json.loads(line)
        except ValueError:
            pass

def load_superpose_timeseries(filenames, atom_indices, topology):
    X = []
    i = []
    f = []
    for file in filenames:
        kwargs = {}  if file.endswith('.h5') else {'top': topology}
        t = md.load(file, **kwargs)
        t.superpose(topology, atom_indices=atom_indices)
        diff2 = (t.xyz[:, atom_indices] - topology.xyz[0, atom_indices])**2
        x = np.sqrt(np.sum(diff2, axis=2))

        X.append(x)
        i.append(np.arange(len(x)))
        f.extend([file]*len(x))

    return np.concatenate(X), np.concatenate(i), np.array(f)

def log_multivariate_normal_pdf(x, mu, Sigma):
    size = len(x)
    if size == len(mu) and (size, size) == shape(Sigma):
      det = linalg.det(Sigma)
      if det == 0:
        raise NameError("The covariance matrix can't be singular")

      try:
        log_norm_const = -0.5 * (float(size) * log(2*pi) + log(det))
      except FloatingPointError:
        log_norm_const = -Inf

      x_mu = x - mu
      inv = linalg.pinv(Sigma)
      log_result = -0.5 * dot(x_mu, dot(inv, x_mu.T))
      return log_norm_const + log_result
    else:
      raise NameError("The dimensions of the input don't match")

def logsumexp(x, dim=-1):
    """Compute log(sum(exp(x))) in a numerically stable way.

       Use second argument to specify along which dimensions the logsumexp
       shall be computed. If -1 (which is the default), logsumexp is
       computed along the last dimension.
    """
    if len(x.shape) < 2:
        xmax = x.max()
        return xmax + log(sum(exp(x-xmax)))
    else:
        if dim != -1:
            x = x.transpose(range(dim) + range(dim+1, len(x.shape)) + [dim])
        lastdim = len(x.shape)-1
        xmax = x.max(lastdim)
        return xmax + log(sum(exp(x-xmax[...,newaxis]),lastdim))

def iter_vars(A, Q,N):
  V = eye(shape(A)[0])
  for i in range(N):
    V = Q + dot(A,dot(V, A.T))
  return V

def assignment_to_weights(assignments,K):
  (T,) = shape(assignments)
  W_i_Ts = zeros((T,K))
  for t in range(T):
    ind = assignments[t]
    for k in range(K):
      if k != ind:
        W_i_Ts[t,k] = 0.0
      else:
        W_i_Ts[t,ind] = 1.0
  return W_i_Ts

def empirical_wells(Ys, W_i_Ts):
  (T, y_dim) = shape(Ys)
  (_, K) = shape(W_i_Ts)
  means = zeros((K, y_dim))
  covars = zeros((K, y_dim, y_dim))
  for k in range(K):
    num = zeros(y_dim)
    denom = 0
    for t in range(T):
      num += W_i_Ts[t, k] * Ys[t]
      denom += W_i_Ts[t,k]
    means[k] = (1.0/denom) * num
  for k in range(K):
    num = zeros((y_dim, y_dim))
    denom = 0
    for t in range(T):
      num += W_i_Ts[t, k] * outer(Ys[t] - means[k], Ys[t] - means[k])
      denom += W_i_Ts[t,k]
    covars[k] = (1.0/denom) * num
  return means, covars

def transition_counts(assignments, K):
  (T,) = shape(assignments)
  Zhat = ones((K, K))
  for t in range(1,T):
    i = assignments[t-1]
    j = assignments[t]
    Zhat[i,j] += 1
  for i in range(K):
    s = sum(Zhat[i])
    Zhat[i] /= s
  return Zhat

def kmeans(ys, K):
  """ Takes a dataset and finds the K means through the usual
  k-means algorithm.
  Inputs:
    ys: Dataset of points
    K: number of means
  Outputs:
    means: Learned means
    assigments: Says which mean the t-th datapoint belongs to
  """
  (T, y_dim) = shape(ys)
  means = zeros((K, y_dim))
  old_means = zeros((K, y_dim))
  assignments = zeros(T)
  num_each = zeros(K)
  # Pick random observations as initializations
  for k in range(K):
    r = randint(0,T)
    means[k] = ys[r]
  Delta = Inf
  Epsilon = 1e-5
  iteration = 0
  while Delta >= Epsilon:
    Delta = 0
    # Perform an Assignment Step
    for t in range(T):
      dist = Inf
      y = ys[t]
      # Find closest means
      for k in range(K):
        if norm(y - means[k]) < dist:
          dist = norm(y - means[k])
          assignments[t] = k
    # Perform Mean Update Step
    old_means[:] = means[:]
    # Reset means and num_each
    means[:] = 0
    num_each[:] = 0
    for t in range(T):
      k = assignments[t]
      num_each[k] += 1
      means[k] += ys[t]
    for k in range(K):
      if num_each[k] > 0:
        means[k] /= num_each[k]
      Delta += norm(means[k] - old_means[k])
    # reset numeach
    iteration += 1
  return means, assignments

def means_match(base_means, means, assignments):
  (K, y_dim) = shape(means)
  (T,) = shape(assignments)
  matching = zeros(K)
  new_assignments = zeros(T)
  for i in range(K):
    closest = -1
    closest_dist = Inf
    for j in range(K):
      if norm(base_means[i] - means[j]) < closest_dist:
        closest = j
        #print "base_means[%d] = %s" % (i, str(base_means[i]))
        #print "means[%d] = %s" % (j, str(means[j]))
        closest_dist = norm(base_means[i]- means[j])
    matching[i] = closest
  for t in range(T):
    new_assignments[t] = matching[assignments[t]]
  return matching, new_assignments
