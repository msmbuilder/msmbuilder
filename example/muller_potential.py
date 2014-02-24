"""Propagating 2D dynamics on the muller potential using OpenMM.

Currently, we just put a harmonic restraint on the z coordinate,
since OpenMM needs to work in 3D. This isn't really a big deal, except
that it affects the meaning of the temperature and kinetic energy. So
take the meaning of those numbers with a grain of salt.
"""
from mixtape.mslds import *
from numpy import array, reshape, savetxt, loadtxt, zeros
from simtk.unit import kelvin, picosecond, femtosecond, nanometer, dalton
from mixtape.utils import *
import simtk.openmm as mm
import matplotlib.pyplot as pp
import numpy as np
import sys
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

class MullerForce(mm.CustomExternalForce):
  """OpenMM custom force for propagation on the Muller Potential. Also
  includes pure python evaluation of the potential energy surface so that
  you can do some plotting"""
  aa = [-1, -1, -6.5, 0.7]
  bb = [0, 0, 11, 0.6]
  cc = [-10, -10, -6.5, 0.7]
  AA = [-200, -100, -170, 15]
  XX = [1, 0, -0.5, -1]
  YY = [0, 0.5, 1.5, 1]

  def __init__(self):
    # start with a harmonic restraint on the Z coordinate
    expression = '1000.0 * z^2'
    for j in range(4):
        # add the muller terms for the X and Y
        fmt = dict(aa=self.aa[j], bb=self.bb[j],
            cc=self.cc[j], AA=self.AA[j],
            XX=self.XX[j], YY=self.YY[j])
        expression += '''+ {AA}*exp({aa} *(x - {XX})^2
                         + {bb} * (x - {XX}) * (y - {YY})
                         + {cc} * (y - {YY})^2)'''.format(**fmt)
    super(MullerForce, self).__init__(expression)

  @classmethod
  def potential(cls, x, y):
    "Compute the potential at a given point x,y"
    value = 0
    for j in range(4):
      try:
        value += cls.AA[j] * np.exp(cls.aa[j] * (x - cls.XX[j])**2 + \
            cls.bb[j] * (x - cls.XX[j]) * (y - cls.YY[j]) +\
            cls.cc[j] * (y - cls.YY[j])**2)
      except FloatingPointError:
        value = np.exp(100)
    return value

  @classmethod
  def plot(cls, ax=None, minx=-1.5, maxx=1.2, miny=-0.2,
    maxy=2, **kwargs):
    "Plot the Muller potential"
    grid_width = max(maxx-minx, maxy-miny) / 200.0
    ax = kwargs.pop('ax', None)
    xx, yy = np.mgrid[minx : maxx : grid_width, miny : maxy : grid_width]
    V = cls.potential(xx, yy)
    # clip off any values greater than 200, since they mess up
    # the color scheme
    if ax is None:
        ax = pp
    ax.contourf(xx, yy, V.clip(max=200), 40, **kwargs)

# Now run code
PLOT = True
LEARN = True
NUM_TRAJS = 1

# each particle is totally independent
nParticles = 1
mass = 1.0 * dalton
#temps  = 200 300 500 750 1000 1250 1500 1750 2000
temperature = 3000 * kelvin
friction = 100 / picosecond
timestep = 10.0 * femtosecond
T = 500
sim_T = 1000

x_dim = 2
y_dim = 2
K = 3
NUM_ITERS = 5

em_vars = ['As', 'bs', 'Qs', 'Z', 'mus']
As = zeros((K, x_dim, x_dim))
bs = zeros((K, x_dim))
mus = zeros((K, x_dim))
Sigmas = zeros((K, x_dim, x_dim))
Qs = zeros((K, x_dim, x_dim))

# Allocate Memory
start = T/4
n_seq = 1
xs = zeros((n_seq, NUM_TRAJS * (T-start), y_dim))

if PLOT:
	# Clear Display
	pp.cla()
# Choose starting conformations uniform on the grid
# between (-1.5, -0.2) and (1.2, 2)
########################################################################

for traj in range(NUM_TRAJS):
  system = mm.System()
  mullerforce = MullerForce()
  for i in range(nParticles):
    system.addParticle(mass)
    mullerforce.addParticle(i, [])
  system.addForce(mullerforce)

  integrator = mm.LangevinIntegrator(temperature, friction, timestep)
  context = mm.Context(system, integrator)
  startingPositions = (np.random.rand(nParticles, 3) * np.array([2.7, 1.8, 1])) + np.array([-1.5, -0.2, 0])

  context.setPositions(startingPositions)
  context.setVelocitiesToTemperature(temperature)

  trajectory = zeros((T,2))
  print "Traj %d" % traj
  for i in range(T):
    x = context.getState(getPositions=True).\
          getPositions(asNumpy=True).value_in_unit(nanometer)
    # Save the state
    if i > start:
      xs[0,traj * (T-start) + (i-start),:] = x[0,0:2]
    trajectory[i,:] = x[0,0:2]
    integrator.step(10)
if LEARN:
  # Compute K-means
  means, assignments = kmeans(xs[0], K)
  W_i_Ts = assignment_to_weights(assignments,K)
  emp_means, emp_covars = empirical_wells(xs[0], W_i_Ts)
  for i in range(K):
    A = randn(x_dim, x_dim)
    u, s, v = np.linalg.svd(A)
    As[i] = 0.5 * rand() * dot(u, v.T)
    bs[i] = dot(eye(x_dim) - As[i], means[i])
    mus[i] = emp_means[i]
    Sigmas[i] = emp_covars[i]
    Qs[i] = 0.5 * Sigmas[i]

  # Learn the MetastableSwitchingLDS
  bs = means
  l = MetastableSwitchingLDS(K, x_dim, n_iter=NUM_ITERS)
  l.fit(xs)
  sim_xs,sim_Ss = l.sample(sim_T,init_state=0, init_obs=means[0])

if PLOT:
  pp.plot(trajectory[start:,0], trajectory[start:,1], color='k')
  # Compute K-means
  means, assignments = kmeans(xs[0], K)
  pp.scatter(means[:,0], means[:,1], color='r',zorder=10)
  pp.scatter(xs[0,:,0], xs[0,:,1], edgecolor='none', facecolor='k',zorder=1)
  Delta = 0.5
  minx = min(xs[0,:,0])
  maxx = max(xs[0,:,0])
  miny = min(xs[0,:,1])
  maxy = max(xs[0,:,1])
  if LEARN:
    minx = min(min(sim_xs[:,0]), minx) - Delta
    maxx = max(max(sim_xs[:,0]), maxx) + Delta
    miny = min(min(sim_xs[:,1]), miny) - Delta
    maxy = max(max(sim_xs[:,1]), maxy) + Delta
    pp.scatter(sim_xs[:,0], sim_xs[:,1], edgecolor='none',
        zorder=5,facecolor='g')
    pp.plot(sim_xs[:,0], sim_xs[:,1], zorder=5,color='g')
  MullerForce.plot(ax=pp.gca(),minx=minx,maxx=maxx,miny=miny,maxy=maxy)
  pp.show()

