from mixtape.mslds import *
from mixtape.ghmm import *
from mixtape.utils import *
from numpy import array, reshape, savetxt, loadtxt, eye
import matplotlib.pyplot as plt
from numpy.random import rand
from numpy.linalg import svd
import sys
import warnings

# Usual
SAMPLE = False
LEARN = True
PLOT = True

# For param changes
# TODO: Make parameter changing automatic
#SAMPLE = True
#LEARN = False
#PLOT = False
