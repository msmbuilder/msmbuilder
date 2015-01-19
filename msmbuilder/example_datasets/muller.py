from __future__ import print_function, division, absolute_import

import time
import numbers
from os import makedirs
from os.path import join
from os.path import exists
from multiprocessing import cpu_count
from multiprocessing.pool import ThreadPool

import numpy as np
from sklearn.utils import check_random_state

from .brownian1d import _NWell
from .base import Bunch, Dataset, get_data_home
from ._muller import propagate, muller_potential

__all__ = ['load_muller', 'MullerPotential']


###############################################################################
# Constants
###############################################################################

# DO NOT CHANGE THESE CONSTANTS WITHOUT UPDATING VERSION ATTRIBUTE
MULLER_PARAMETERS = dict(
    MIN_X=-1.5, MIN_Y=-0.2,
    MAX_X=1.2, MAX_Y=2,
    N_TRAJECTORIES=10,
    N_STEPS=1000000,
    THIN=100,
    KT=1.5e4,
    DT=0.1,
    DIFFUSION_CONST=1e-2,
    VERSION=1)

MULLER_DESCRIPTION = """Brownian dynamics on the 2D Muller potential
This dataset consists of {N_TRAJECTORIES} trajectories simulated with Brownian
dynamics on the Muller potential, a two-dimensional, three-well potential
energy surface. The potential is defined in [1].
The dynamics governed by the stochastic differential equation
    dx_t/dt = -\nabla V(x)/(kT) + \sqrt{{2D}} * R(t)
where R(t) is a standard normal white-noise process, and D={DIFFUSION_CONST}.
The timsetep is dt={DT}, and kT={KT} Each trajectory is simulated from
{N_STEPS}, and coordinates are saved every {THIN} steps. The starting points
for the trajectories are sampled from the uniform distribution over the
rectangular box between x=({MIN_X}, {MAX_X}) and y=(({MIN_Y}, {MAX_Y}).

References
----------
.. [1] Muller, Klaus, and Leo D. Brown. "Location of saddle points and minimum
   energypaths by a constrained simplex optimization procedure." Theoretica
   chimica acta 53.1 (1979): 75-93.
""".format(**MULLER_PARAMETERS)


###############################################################################
# Code
###############################################################################

class MullerPotential(_NWell):
    """Browian dynamics on Muller potential dataset

    Parameters
    ----------
    data_home : optional, default: None
        Specify another cache folder for the datasets. By default
        all mixtape data is stored in '~/mixtape_data' subfolders.
    random_state : {int, None}, default: None
        Seed the psuedorandom number generator to generate trajectories. If
        seed is None, the global numpy PRNG is used. If random_state is an
        int, the simulations will be cached in ``data_home``, or loaded from
        ``data_home`` if simulations with that seed have been performed
        already. With random_state=None, new simulations will be performed and
        the trajectories will not be cached.

    Notes
    -----
    """
    target_name = "muller"
    version = MULLER_PARAMETERS['VERSION']

    def simulate_func(self, random):
        M = MULLER_PARAMETERS
        x0 = random.uniform(
            low=[M['MIN_X'], M['MIN_Y']],
            high=[M['MAX_X'], M['MAX_Y']],
            size=(M['N_TRAJECTORIES'], 2))

        # propagate releases the GIL, so we can use a thread pool and
        # get a nice speedup
        tp = ThreadPool(cpu_count())
        return tp.map(lambda x0:
            propagate(
                n_steps=M['N_STEPS'], x0=x0, thin=M['THIN'], kT=M['KT'],
                dt=M['DT'], D=M['DIFFUSION_CONST'], random_state=random), x0)

    def potential(self, x, y):
        return muller_potential(x, y)

    def plot(self, minx=-1.5, maxx=1.2, miny=-0.2, maxy=2, **kwargs):
        """Helper function to plot the Muller potential
        """
        import matplotlib.pyplot as pp
        grid_width = max(maxx-minx, maxy-miny) / 200.0

        ax = kwargs.pop('ax', None)

        xx, yy = np.mgrid[minx:maxx:grid_width, miny:maxy:grid_width]
        V = self.potential(xx, yy)
        # clip off any values greater than 200, since they mess up
        # the color scheme
        if ax is None:
            ax = pp

        ax.contourf(xx, yy, V.clip(max=200), 40, **kwargs)


def load_muller(data_home=None, random_state=None):
    return MullerPotential(data_home, random_state).get()

MullerPotential.__doc__ += MULLER_DESCRIPTION
load_muller.__doc__ = MullerPotential.__doc__
