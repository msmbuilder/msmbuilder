"""Base IO code for all datasets
"""

# Copyright (c) 2007 David Cournapeau <cournape@gmail.com>
# 2010 Fabian Pedregosa <fabian.pedregosa@inria.fr>
# 2010 Olivier Grisel <olivier.grisel@ensta.org>
# License: BSD 3 clause
# Adapted for msmbuilder from https://github.com/scikit-learn/scikit-learn/blob/master/sklearn/datasets/base.py
import warnings

try:
    from msmb_data.base import (Bunch, Dataset, get_data_home,
                                clear_data_home, retry)
except ImportError:
    warnings.warn("Please install msmb_data", DeprecationWarning)
    from .old_base import (Bunch, Dataset, get_data_home,
                                    clear_data_home, retry)
