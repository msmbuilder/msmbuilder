"""Very simple datasets of brownian dynamics in one dimension.
"""
DOUBLEWELL_DESCRIPTION=r"""Brownian dynamics on a 1D double well potential

This dataset consists of 10 trajectories simulated with Brownian dynamics on
the reduced potential function:

    V(x) = 1 + cos(2x)

with reflecting boundary conditions at x=-pi and x=pi. The simulations are
governed by the stochastic differential equation

    dx_t/dt = -\nabla V(x) + \sqrt{2D} * R(t)

where R(t) is a standard normal white-noise process, and D=1e3. The timsetep
is 1e-3. Each trajectory is 10^5 steps long, and starts at x_0 = 0.
"""

#-----------------------------------------------------------------------------
# Imports
#-----------------------------------------------------------------------------

import time
import numbers
from os import makedirs
from os.path import join
from os.path import exists
import numpy as np
from sklearn.utils import check_random_state
from mixtape.utils import verboseload, verbosedump

from mixtape.datasets.base import Bunch
from mixtape.datasets.base import get_data_home

#-----------------------------------------------------------------------------
# Globals
#-----------------------------------------------------------------------------

# DO NOT CHANGE THESE CONSTANTS WITHOUT UPDATING THE
# "DOUBLEWELL_DESCRIPTION" VARIABLE
DIFFUSION_CONST = 1e3
DT = 1e-3
DT_SQRT_2D = DT * np.sqrt(2 * DIFFUSION_CONST)


#-----------------------------------------------------------------------------
# User functions
#-----------------------------------------------------------------------------

def load_doublewell(data_home=None, random_state=None):
    """Loader for double-well dataset

    Parameters
    ----------
    data_home : optional, default: None
        Specify another cache folder for the datasets. By default
        all mixtape data is stored in '~/mixtape_data' subfolders.
    random_state : {int, None}, default: None
        Seed the psuedorandom number generator to generate trajectories. If
        seed is None, the global numpy PRNG is used. If random_state is an
        int, the simulations will be cached in ``data_home``, or loaded from
        ``data_home`` if simulations with that seed have been performed already.
        With random_state=None, new simulations will be performed and the
        trajectories will not be cached.

    Notes
    -----
    """
    random = check_random_state(random_state)
    data_home = join(get_data_home(data_home=data_home), 'doublewell')
    if not exists(data_home):
        makedirs(data_home)

    if random_state is None:
        trajectories = _simulate_doublewell(random)
    else:
        assert isinstance(random_state, numbers.Integral), 'random_state but be an int'
        path = join(data_home, 'version-1_random-state-%d.pkl' % random_state)
        if exists(path):
            trajectories = verboseload(path)
        else:
            trajectories = _simulate_doublewell(random)
            verbosedump(trajectories, path)

    return Bunch(trajectories=trajectories, DESCR=DOUBLEWELL_DESCRIPTION)

load_doublewell.__doc__ += DOUBLEWELL_DESCRIPTION

#-----------------------------------------------------------------------------
# Internal functions
#-----------------------------------------------------------------------------

DOUBLEWELL_GRAD_POTENTIAL = lambda x : -2 * np.sin(2 * x)

def _simulate_doublewell(random):
    # DO NOT CHANGE THESE CONSTANTS WITHOUT UPDATING THE
    # "DOUBLEWELL_DESCRIPTION" VARIABLE
    x0 = 0
    n_steps = 1e5
    n_trajectories = 10

    trajectories = [_propagate1d(x0, n_steps, DOUBLEWELL_GRAD_POTENTIAL, random,
                                 bc_min=-np.pi, bc_max=np.pi).reshape(-1, 1)
                    for i in range(n_trajectories)]
    return trajectories

def _reflect_boundary_conditions(x, min, max):
    if x > max:
        return 2*max - x
    if x < min:
        return 2*min - x
    return x


def _propagate1d(x0, n_steps, grad_potential, random, bc_min=None, bc_max=None):
    start = time.time()
    n_steps = int(n_steps)

    if bc_min is None and bc_max is None:
        bc = lambda x : x
    else:
        bc = lambda x: _reflect_boundary_conditions(x, bc_min, bc_max)

    rand = random.randn(n_steps)
    x = np.zeros(n_steps+1)
    x[0] = x0
    for i in range(n_steps):
        x_i_plus_1 = x[i] -DT * grad_potential(x[i]) + DT_SQRT_2D*rand[i]
        x[i+1] = bc(x_i_plus_1)

    print('%d steps/s' % (n_steps / (time.time() - start)))
    return x


def doublewell_eigs(n_grid, lag_time=1, grad_potential=DOUBLEWELL_GRAD_POTENTIAL):
    """Analytic eigenvalues/eigenvectors for the doublwell system.

    """
    import scipy.linalg
    ONE_OVER_SQRT_2PI = 1.0 / (np.sqrt(2*np.pi))
    normalpdf = lambda x : ONE_OVER_SQRT_2PI * np.exp(-0.5 * (x*x))

    grid = np.linspace(-np.pi, np.pi, n_grid)
    width = grid[1]-grid[0]
    transmat = np.zeros((n_grid, n_grid))
    for i, x_i in enumerate(grid):
        for offset in range(-(n_grid-1), n_grid):
            x_j = x_i + (offset * width)
            j = _reflect_boundary_conditions(i+offset, 0, n_grid-1)

            # What is the probability of going from x_i to x_j in one step?
            diff = (x_j - x_i + DT * grad_potential(x_i)) / DT_SQRT_2D
            transmat[i, j] += normalpdf(diff)
        transmat[i, :] =  transmat[i, :] / np.sum(transmat[i, :])


    transmat = np.linalg.matrix_power(transmat, lag_time)
    u, v = scipy.linalg.eig(transmat)
    order = np.argsort(np.real(u))[::-1]

    return np.real_if_close(u[order]), np.real_if_close(v[:, order])
