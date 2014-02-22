from mixtape.mslds import *
from mixtape.utils import *
from numpy import array, reshape, savetxt, loadtxt
import matplotlib.pyplot as plt
from numpy.random import rand
from numpy.linalg import svd
import sys

"""The switching system has the following one-dimensional dynamics:
    x_{t+1}^1 = x_t + \epsilon_1
    x_{t+1}^2 = -x_t + \epsilon_2
"""
# Usual
SAMPLE = False
LEARN = True
PLOT = False

## For param changes
#SAMPLE = True
#LEARN = False
#PLOT = False

NUM_ITERS = 10
T = 500
x_dim = 1
y_dim = 1
K = 2
As = reshape(array([[0.5],[0.5]]), (K,x_dim,x_dim))
bs = reshape(array([[0.5],[-0.5]]), (K,x_dim))
Qs = reshape(array([[0.01],[0.01]]), (K,x_dim,x_dim))
Z = reshape(array([[0.98, 0.02],[0.02, 0.98]]), (K,K))
pi = reshape(array([0.99,0.01]), (K,))
mus = reshape(array([[1],[-1]]), (K,x_dim))
Sigmas = reshape(array([[0.01],[0.01]]), (K,x_dim,x_dim))

em_vars = ['As', 'bs', 'Qs', 'Z', 'mus', 'Sigmas']
s = MetastableSwitchingLDS(x_dim,y_dim,K=K,As=As,bs=bs,Qs=Qs,Z=Z)
if SAMPLE:
  xs,Ss = s.sample(T)
  savetxt('../example/xs.txt', xs)
  savetxt('../example/Ss.txt', Ss)
else:
  xs = reshape(loadtxt('../example/xs.txt'), (T,x_dim))
  Ss = reshape(loadtxt('../example/Ss.txt'), (T))

if LEARN:
  As = zeros((K, x_dim, x_dim))
  bs = zeros((K, x_dim))
  mus = zeros((K, x_dim))
  Sigmas = zeros((K, x_dim, x_dim))
  Qs = zeros((K, x_dim, x_dim))
  # Compute K-means
  means, assignments = kmeans(xs, K)
  W_i_Ts = assignment_to_weights(assignments,K)
  emp_means, emp_covars = empirical_wells(xs, W_i_Ts)
  for i in range(K):
    A = randn(x_dim, x_dim)
    u, s, v = svd(A)
    As[i] = 0.5 * rand() * dot(u, v.T)
    bs[i] = dot(eye(x_dim) - As[i], means[i])
    mus[i] = emp_means[i]
    Sigmas[i] = emp_covars[i]
    Qs[i] = 0.5 * Sigmas[i]
  l = MetastableSwitchingLDS(x_dim,y_dim,K=K,
      As=As,bs=bs,mus=mus,Sigmas=Sigmas,Qs=Qs)
  l.em(xs, em_iters=NUM_ITERS, em_vars=em_vars)
  sim_xs,sim_Ss = l.sample(T,s_init=0, x_init=means[0],
      y_init=means[0])

if PLOT:
  plt.close('all')
  plt.figure(1)
  plt.plot(range(T), xs, label="Observations")
  if LEARN:
    plt.plot(range(T), sim_xs, label='Sampled Observations')
  plt.legend()
  plt.show()
