"""Very simple datasets of brownian dynamics in one dimension."""
# Author: Robert McGibbon <rmcgibbo@gmail.com>
# Contributors:
# Copyright (c) 2014, Stanford University
# All rights reserved.

# -----------------------------------------------------------------------------
# Imports
# -----------------------------------------------------------------------------
from __future__ import absolute_import

import time

import numpy as np

from .base import _NWell
from ..msm import _solve_msm_eigensystem

# -----------------------------------------------------------------------------
# Globals
# -----------------------------------------------------------------------------

# DO NOT CHANGE THESE CONSTANTS WITHOUT UPDATING THE
# "DOUBLEWELL_DESCRIPTION" VARIABLE
DIFFUSION_CONST = 1e3
DT = 1e-3
DT_SQRT_2D = DT * np.sqrt(2 * DIFFUSION_CONST)

__all__ = ['load_doublewell', 'load_quadwell',
           'doublewell_eigs', 'quadwell_eigs']


# -----------------------------------------------------------------------------
# User functions
# -----------------------------------------------------------------------------



class DoubleWell(_NWell):
    r"""Brownian dynamics on a 1D double well potential

    Parameters
    ----------
    data_home : optional, default: None
        Specify another cache folder for the datasets. By default
        all MSMBuilder data is stored in '~/msmbuilder_data' subfolders.
    random_state : {int, None}, default: None
        Seed the psuedorandom number generator to generate trajectories. If
        seed is None, the global numpy PRNG is used. If random_state is an
        int, the simulations will be cached in ``data_home``, or loaded from
        ``data_home`` if simulations with that seed have been performed already.
        With random_state=None, new simulations will be performed and the
        trajectories will not be cached.

    Notes
    -----
    This dataset consists of 10 trajectories simulated with Brownian dynamics on
    the reduced potential function

        V(x) = 1 + cos(2x)

    with reflecting boundary conditions at x=-pi and x=pi. The simulations
    are governed by the stochastic differential equation

        dx_t/dt = -\nabla V(x) + \sqrt{2D} * R(t),

    where R(t) is a standard normal white-noise process, and D=1e3. The
    timsetep is 1e-3. Each trajectory is 10^5 steps long, and starts at
    x_0 = 0.
    """
    target_name = "doublewell"
    n_trajectories = 10

    def simulate_func(self, random):
        return _simulate_doublewell(random)

    def potential(self, x):
        return 1 + np.cos(2 * x)


def load_doublewell(data_home=None, random_state=None):
    return DoubleWell(data_home, random_state).get()


load_doublewell.__doc__ = DoubleWell.__doc__


class QuadWell(_NWell):
    r"""Brownian dynamics on a 1D four well potential

    Parameters
    ----------
    data_home : optional, default: None
        Specify another cache folder for the datasets. By default
        all MSMBuilder data is stored in '~/msmbuilder_data' subfolders.
    random_state : {int, None}, default: None
        Seed the psuedorandom number generator to generate trajectories. If
        seed is None, the global numpy PRNG is used. If random_state is an
        int, the simulations will be cached in ``data_home``, or loaded from
        ``data_home`` if simulations with that seed have been performed already.
        With random_state=None, new simulations will be performed and the
        trajectories will not be cached.

    Notes
    -----
    This dataset consists of 100 trajectories simulated with Brownian dynamics
    on the reduced potential function

    V = 4(x^8 + 0.8 exp(-80 x^2) + 0.2 exp(-80 (x-0.5)^2) + 0.5 exp(-40 (x+0.5)^2)).

    The simulations are governed by the stochastic differential equation

    dx_t/dt = -\nabla V(x) + \sqrt{2D} * R(t),

    where R(t) is a standard normal white-noise process, and D=1e3. The timsetep
    is 1e-3. Each trajectory is 10^3 steps long, and starts from a random
    initial point sampled from the uniform distribution on [-1, 1].
    """

    target_name = "quadwell"
    n_trajectories = 100

    def simulate_func(self, random):
        return _simulate_quadwell(random)

    def potential(self, x):
        return 4 * (x ** 8 + 0.8 * np.exp(-80 * x ** 2) + 0.2 * np.exp(
            -80 * (x - 0.5) ** 2) +
                    0.5 * np.exp(-40 * (x + 0.5) ** 2))


def load_quadwell(data_home=None, random_state=None):
    return QuadWell(data_home, random_state).get()


load_quadwell.__doc__ = QuadWell.__doc__


def doublewell_eigs(n_grid, lag_time=1):
    """Analytic eigenvalues/eigenvectors for the doublwell system

    TODO: DOCUMENT ME
    """
    return _brownian_eigs(n_grid, lag_time, DOUBLEWELL_GRAD_POTENTIAL,
                          -np.pi, np.pi, reflect_bc=True)


def quadwell_eigs(n_grid, lag_time=1):
    """Analytic eigenvalues/eigenvectors for the quadwell system

    TODO: DOCUMENT ME
    """
    return _brownian_eigs(n_grid, lag_time, QUADWELL_GRAD_POTENTIAL,
                          -1.2, 1.2, reflect_bc=False)


# -----------------------------------------------------------------------------
# Internal functions
# -----------------------------------------------------------------------------

DOUBLEWELL_GRAD_POTENTIAL = lambda x: -2 * np.sin(2 * x)
QUADWELL_GRAD_POTENTIAL = lambda x: 4 * (
    8 * x ** 7 - 128 * x * np.exp(-80 * x ** 2) - \
    32 * (x - 0.5) * np.exp(-80 * (x - 0.5) ** 2) - 40 * (x + 0.5) * np.exp(
        -40 * (x + 0.5) ** 2))


def _simulate_doublewell(random):
    # DO NOT CHANGE THESE CONSTANTS WITHOUT UPDATING THE
    # "DOUBLEWELL_DESCRIPTION" VARIABLE AND UPDATING THE VERSION NUMBER
    # in the load_doublewell FUNCTION
    x0 = 0
    n_steps = 1e5
    n_trajectories = 10

    trajectories = [_propagate1d(
        x0, n_steps, DOUBLEWELL_GRAD_POTENTIAL, random, bc_min=-np.pi,
        bc_max=np.pi, verbose=True).reshape(-1, 1)
                    for i in range(n_trajectories)]
    return trajectories


def _simulate_quadwell(random):
    # DO NOT CHANGE THESE CONSTANTS WITHOUT UPDATING THE
    # "QUADWELL_DESCRIPTION" VARIABLE AND UPDATING THE VERSION NUMBER
    # in the load_quadwell FUNCTION
    n_steps = 1e3
    n_trajectories = 100
    x0 = random.uniform(-1, 1, size=n_trajectories)

    trajectories = [_propagate1d(
        x0[i], n_steps, QUADWELL_GRAD_POTENTIAL,
        random=random, verbose=False).reshape(-1, 1)
                    for i in range(n_trajectories)]
    return trajectories


def _reflect_boundary_conditions(x, min, max):
    if x > max:
        return 2 * max - x
    if x < min:
        return 2 * min - x
    return x


def _propagate1d(x0, n_steps, grad_potential, random, bc_min=None, bc_max=None,
                 verbose=True):
    start = time.time()
    n_steps = int(n_steps)

    if bc_min is None and bc_max is None:
        bc = lambda x: x
    else:
        bc = lambda x: _reflect_boundary_conditions(x, bc_min, bc_max)

    rand = random.randn(n_steps)
    x = np.zeros(n_steps + 1)
    x[0] = x0
    for i in range(n_steps):
        x_i_plus_1 = x[i] - DT * grad_potential(x[i]) + DT_SQRT_2D * rand[i]
        x[i + 1] = bc(x_i_plus_1)

    if verbose:
        print('%d steps/s' % (n_steps / (time.time() - start)))
    return x


def _brownian_eigs(n_grid, lag_time, grad_potential, xmin, xmax, reflect_bc):
    """Analytic eigenvalues/eigenvectors for 1D Brownian dynamics
    """

    ONE_OVER_SQRT_2PI = 1.0 / (np.sqrt(2 * np.pi))
    normalpdf = lambda x: ONE_OVER_SQRT_2PI * np.exp(-0.5 * (x * x))

    grid = np.linspace(xmin, xmax, n_grid)
    width = grid[1] - grid[0]
    transmat = np.zeros((n_grid, n_grid))
    for i, x_i in enumerate(grid):
        if reflect_bc:
            for offset in range(-(n_grid - 1), n_grid):
                x_j = x_i + (offset * width)
                j = _reflect_boundary_conditions(i + offset, 0, n_grid - 1)

                # What is the probability of going from x_i to x_j in one step?
                diff = (x_j - x_i + DT * grad_potential(x_i)) / DT_SQRT_2D
                transmat[i, j] += normalpdf(diff)
        else:
            for j, x_j in enumerate(grid):
                # What is the probability of going from x_i to x_j in one step?
                diff = (x_j - x_i + DT * grad_potential(x_i)) / DT_SQRT_2D
                transmat[i, j] += normalpdf(diff)

        transmat[i, :] = transmat[i, :] / np.sum(transmat[i, :])

    transmat = np.linalg.matrix_power(transmat, lag_time)
    u, lv, rv = _solve_msm_eigensystem(transmat, k=len(transmat) - 1)
    return u, rv
