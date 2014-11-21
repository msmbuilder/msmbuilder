import warnings

warnings.warn("Please use the module msm instead of markovstatemodel. " +
              "This alias will be removed in version 3.1", DeprecationWarning)

from .msm import *
