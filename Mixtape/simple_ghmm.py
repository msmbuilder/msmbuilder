"""
Hidden Markov Models in Python.

This code is adapted in large part from code released by Roland Memisevic.
"""


from numpy import sum, zeros, ones, newaxis, pi, exp, log, dot, eye, diag, arccos, sqrt, array, vstack, prod, isfinite, inf, real, arange, sin, cos, hstack, where, mean, double, concatenate, shape
from numpy import nonzero
from numpy.linalg.linalg import svd, eigh
import numpy.random
from pylab import randn, scatter, hold, gca, axes, cla, norm, plot, imshow, cm
from matplotlib.patches import Ellipse

log2pi = log(2) + log(pi)

def absdet(M):
    U,D,Vt = svd(M)
    wellConditioned = D>0.000000001
    return prod(D[wellConditioned])


def pinv(M):
    U,D,Vt = svd(M)
    wellConditioned = D>0.000000001
    return dot(U[:,wellConditioned],
               dot(diag(D[wellConditioned]**-1.0), Vt[wellConditioned,:]))

def plotGaussian(m, covar):
    """ plot a 2d gaussian """

    t = arange(-pi,pi,0.01)
    k = len(t)
    x = sin(t)[:, newaxis]
    y = cos(t)[:, newaxis]

    D, V = eigh(covar)
    A = real(dot(V,diag(sqrt(D))).T)
    z = dot(hstack([x, y]), A)

    hold('on')
    plot(z[:,0]+m[0], z[:,1]+m[1])
    plot(array([m[0]]), array([m[1]]))


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

class GaussianHmm(object):
  """ Hidden Markov model with Gaussian observables."""
  def __init__(self,numstates, numdims):
    self.numstates = numstates
    self.numdims = numdims
    self.numparams = self.numdims * self.numstates\
        + self.numdims**2 * self.numstates\
        + self.numstates\
        + self.numstates**2
    self.params = zeros(self.numparams, dtype=float)
    self.means = self.params[:self.numdims*self.numstates].reshape(
          self.numdims,self.numstates)
    self.covs = self.params[self.numdims * self.numstates:
                  self.numdims*self.numstates+
                  self.numdims**2 * self.numstates].reshape(
                      self.numdims, self.numdims, self.numstates)
    self.logInitProbs = self.params[self.numdims * self.numstates +
                          self.numdims**2 * self.numstates:
                          self.numdims * self.numstates +
                          self.numdims**2 * self.numstates+
                          self.numstates]
    self.logTransitionProbs = self.params[-self.numstates**2:].reshape(
                                self.numstates, self.numstates)
    self.means[:] = 0.1 * randn(self.numdims, self.numstates)
    for k in range(self.numstates):
      self.covs[:,:,k] = eye(self.numdims) * 0.1;
    self.logInitProbs[:] = log(ones(self.numstates, dtype=float) /
                              self.numstates)
    self.logTransitionProbs[:] = log(ones((self.numstates,
                                          self.numstates), dtype=float) /
                                    self.numstates)

  def _loggaussian(self, datapoint, k):
    lognormalizer = -0.5 * self.numdims * log2pi \
                    - 0.5 * log(absdet(self.covs[:,:,k]))
    datapoint_m = (datapoint - self.means[:,k])[:, newaxis]
    Kinv = pinv(self.covs[:,:,k])
    return lognormalizer - 0.5 * (dot(datapoint_m.T, dot(Kinv,
      datapoint_m)))

  def _gaussian(self, datapoint, k):
    return exp(self._loggaussian(datapoint, k))

  def alphabeta(self, data):
    """Generates log-alpha and long-beta tables with the alpha-beta
    algorithm."""
    numdims, numpoints = data.shape
    logAlpha = zeros((self.numstates, numpoints), dtype=float)
    logBeta = zeros((self.numstates, numpoints), dtype=float)
    for k in range(self.numstates):
      logAlpha[k,0] = self.logInitProbs[k] + \
          self._loggaussian(data[:,0],k)
      logBeta[k,-1] = 0.0
    for t in range(numpoints - 1):
      for k in range(self.numstates):
        logAlpha[k, t+1] = logsumexp(self.logTransitionProbs[:,k] +\
            logAlpha[:,t]) + self._loggaussian(data[:,t+1],k)
        # necessary every time tick?
        assert isfinite(sum(exp(logAlpha)))
    for t in range(numpoints-2,0,-1):
      for k in range(self.numstates):
        logBeta[k,t] = logsumexp(logBeta[:,t+1] +\
                        self.logTransitionProbs[k,:] +\
          array([self._loggaussian(data[:,t+1], i)
            for i in range(self.numstates)]).flatten())
    return logAlpha, logBeta

  def learn(self, data, numsteps, visualize=False):
    if type(data) is not type([]): # learn on a single sequence
      numdims, numpoints = data.shape
      assert numdims == self.numdims
      logXi = zeros((numpoints-1,self.numstates,self.numstates),
          dtype=float)
      lastlogprob = -inf
      for iteration in range(numsteps):
        print "EM iteration: %d" % iteration

        # E-step
        logAlpha, logBeta = self.alphabeta(data)
        # compute xi and gamma
        for t in range(numpoints-1):
          for i in range(self.numstates):
            for j in range(self.numstates):
              logXi[t,i,j] = logAlpha[i,t] +\
                  self.logTransitionProbs[i,j] +\
                  self._loggaussian(data[:,t+1],j) +\
                  logBeta[j,t+1]
          logXi[t,:,:] -= logsumexp(logXi[t,:,:].flatten())
        logGamma = vstack((logsumexp(logXi, 2),
                           logsumexp(logXi[-1,:,:], 1)))
        logprob = logsumexp(logAlpha[:,-1])

        print "logprob = %f" % logprob
        if abs(logprob - lastlogprob) <= 10**-6:
          print "converged"
          break
        lastlogprob = logprob

        # M-step
        self.logInitProbs[:] = logGamma[0,:]
        self.logTransitionProbs[:] = logsumexp(logXi, 0) - \
            logsumexp(logGamma[:-1,:],0)[:,newaxis]
        G = exp(logGamma - logsumexp(logGamma, 0)[newaxis,:])
        for k in range(self.numstates):
          self.means[:,k] = sum(G[:,k][newaxis,:]*data,1)
          data_m = data - self.means[:,k][:,newaxis]
          self.covs[:,:,k] = dot((data_m*G[:,k][newaxis,:]), data_m.T)

        # threshold eigenvalues
        for k in range(self.numstates):
          U, D, Vt = svd(self.covs[:,:,k])
          D[D<0.01] = 0.01
          self.covs[:,:,k] = dot(U, dot(diag(D), Vt))

        # vsiualize:
        if visualize and self.numdims == 2:
          cla()
          scatter(data[0,:], data[1,:])
          for k in range(self.numstates):
            plotGaussian(self.means[:,k], self.covs[:,:,k])

  def setLogTransitionProbs(self, newLogTransitionProbs):
    self.logTransitionProbs = newLogTransitionProbs

  def setLogInitProbs(self, newLogInitProbs):
    self.logInitProbs = newLogInitProbs

  def setMeans(self, newMeans):
    self.means = newMeans

  def setCovs(self, newCovs):
    self.covs = newCovs

  def generate_data(self, numsteps, initial_state):
    states = zeros(numsteps)
    outputs = zeros((self.numdims, numsteps))
    states[0] = initial_state
    print states
    for t in range(numsteps):
      k = states[t]
      print k
      if t < numsteps - 1:
        logTransitions = self.logTransitionProbs[k,:] - \
            logsumexp(self.logTransitionProbs[k,:])
        print exp(logTransitions)
        sample = numpy.random.multinomial(1,exp(logTransitions))
        next_state = nonzero(sample)[0] # Get sampled index
        print next_state
        states[t+1] = next_state
      mean = self.means[:,k]
      cov = self.covs[:,:,k]
      outputs[:,t] = numpy.random.multivariate_normal(mean, cov)
    return states, outputs
