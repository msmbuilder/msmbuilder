import warnings

warnings.warn("Please use the module decomposition instead of tica. " +
              "This alias will be removed in version 3.1", DeprecationWarning)

from .decomposition import *
