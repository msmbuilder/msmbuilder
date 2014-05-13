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
    #LEARN = True



    #As = zeros((K, x_dim, x_dim))
    #bs = zeros((K, x_dim))
    #mus = zeros((K, x_dim))
    #Sigmas = zeros((K, x_dim, x_dim))
    #Qs = zeros((K, x_dim, x_dim))

    ## Allocate Memory
    #start = T / 4
    #n_seq = 1
    #xs = zeros((n_seq, NUM_TRAJS * (T - start), y_dim))

    ########################################################################

except:
    type, value, tb = sys.exc_info()
    traceback.print_exc()
    pdb.post_mortem(tb)
