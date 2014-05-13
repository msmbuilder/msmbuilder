"""Propagating 2D dynamics on the muller potential using OpenMM.

Currently, we just put a harmonic restraint on the z coordinate,
since OpenMM needs to work in 3D. This isn't really a big deal, except
that it affects the meaning of the temperature and kinetic energy. So
take the meaning of those numbers with a grain of salt.
"""
from mixtape.mslds import *
from mixtape.ghmm import *
from numpy import array, reshape, savetxt, loadtxt, zeros
from simtk.unit import kelvin, picosecond, femtosecond, nanometer, dalton
from mixtape.utils import *
import simtk.openmm as mm
import matplotlib.pyplot as pp
import numpy as np
import sys
import warnings
import traceback, sys, code, pdb
warnings.filterwarnings("ignore", category=DeprecationWarning)

try:
    # Now run code
    PLOT = True
    LEARN = True
    NUM_TRAJS = 1

    # each particle is totally independent
    nParticles = 1
    mass = 1.0 * dalton
    # temps  = 200 300 500 750 1000 1250 1500 1750 2000
    temperature = 500 * kelvin
    friction = 100 / picosecond
    timestep = 10.0 * femtosecond
    T = 2500
    sim_T = 1000

    x_dim = 2
    y_dim = 2
    K = 3
    NUM_HOTSTART = 5
    NUM_ITERS = 10
    MAX_ITERS = 20

    As = zeros((K, x_dim, x_dim))
    bs = zeros((K, x_dim))
    mus = zeros((K, x_dim))
    Sigmas = zeros((K, x_dim, x_dim))
    Qs = zeros((K, x_dim, x_dim))

    # Allocate Memory
    start = T / 4
    n_seq = 1
    xs = zeros((n_seq, NUM_TRAJS * (T - start), y_dim))

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
        startingPositions = (np.random.rand(
            nParticles, 3) * np.array([2.7, 1.8, 1])) + np.array([-1.5, -0.2, 0])

        context.setPositions(startingPositions)
        context.setVelocitiesToTemperature(temperature)

        trajectory = zeros((T, 2))
        for i in range(T):
            x = context.getState(getPositions=True).\
                getPositions(asNumpy=True).value_in_unit(nanometer)
            # Save the state
            if i > start:
                xs[0, traj * (T-start) + (i-start), :] = x[0, 0:2]
            trajectory[i, :] = x[0, 0:2]
            integrator.step(10)
    if LEARN:

    if PLOT:
        if LEARN:
except:
    type, value, tb = sys.exc_info()
    traceback.print_exc()
    pdb.post_mortem(tb)
