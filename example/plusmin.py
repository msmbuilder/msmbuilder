from mixtape.mslds import *
from mixtape.utils import *
from numpy import array, reshape, savetxt, loadtxt
import matplotlib.pyplot as plt
from numpy.random import rand
from numpy.linalg import svd
import sys
import warnings

"""The switching system has the following one-dimensional dynamics:
    x_{t+1}^1 = x_t + \epsilon_1
    x_{t+1}^2 = -x_t + \epsilon_2
"""
warnings.filterwarnings("ignore", category=DeprecationWarning)
# Usual
SAMPLE = False
LEARN = True
PLOT = True

## For param changes
#SAMPLE = True
#LEARN = False
#PLOT = False

n_seq = 1
NUM_ITERS = 10
T = 500
x_dim = 1
K = 2
As = reshape(array([[0.5],[0.5]]), (K,x_dim,x_dim))
bs = reshape(array([[0.5],[-0.5]]), (K,x_dim))
Qs = reshape(array([[0.01],[0.01]]), (K,x_dim,x_dim))
Z = reshape(array([[0.98, 0.02],[0.02, 0.98]]), (K,K))
pi = reshape(array([0.99,0.01]), (K,))
mus = reshape(array([[1],[-1]]), (K,x_dim))
Sigmas = reshape(array([[0.01],[0.01]]), (K,x_dim,x_dim))

#em_vars = ['As', 'bs', 'Qs', 'Z', 'mus', 'Sigmas']
s = MetastableSwitchingLDS(K,x_dim)
s.As_ = As
s.bs_ = bs
s.Qs_ = Qs
s.transmat_ = Z
s.populations_ = pi
s.means_ = mus
s.covars_ = Sigmas
if SAMPLE:
  xs,Ss = s.sample(T)
  xs = reshape(xs, (n_seq, T, x_dim))
  savetxt('./example/xs.txt', xs)
  savetxt('./example/Ss.txt', Ss)
else:
  xs = reshape(loadtxt('./example/xs.txt'), (n_seq, T,x_dim))
  Ss = reshape(loadtxt('./example/Ss.txt'), (n_seq, T))

if LEARN:
  As = zeros((K, x_dim, x_dim))
  bs = zeros((K, x_dim))
  mus = zeros((K, x_dim))
  Sigmas = zeros((K, x_dim, x_dim))
  Qs = zeros((K, x_dim, x_dim))
  # Compute K-means
  print shape(xs[0])
  means, assignments = kmeans(xs[0], K)
  W_i_Ts = assignment_to_weights(assignments,K)
  emp_means, emp_covars = empirical_wells(xs[0], W_i_Ts)
  for i in range(K):
    A = randn(x_dim, x_dim)
    u, s, v = svd(A)
    As[i] = 0.5 * rand() * dot(u, v.T)
    bs[i] = dot(eye(x_dim) - As[i], means[i])
    mus[i] = emp_means[i]
    Sigmas[i] = emp_covars[i]
    Qs[i] = 0.5 * Sigmas[i]
  l = MetastableSwitchingLDS(K,x_dim, n_iter=NUM_ITERS)
  l.fit(xs)
  sim_xs,sim_Ss = l.sample(T,init_state=0, init_obs=means[0])
  sim_xs = reshape(sim_xs, (n_seq, T, x_dim))

if PLOT:
  plt.close('all')
  plt.figure(1)
  plt.plot(range(T), xs[0], label="Observations")
  if LEARN:
    plt.plot(range(T), sim_xs[0], label='Sampled Observations')
  plt.legend()
  plt.show()
