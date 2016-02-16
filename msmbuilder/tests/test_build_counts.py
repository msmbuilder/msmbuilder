from __future__ import print_function, division

from msmbuilder.msm import MarkovStateModel,BayesianMarkovStateModel,\
    ContinuousTimeMSM

from mdtraj.testing import eq
import numpy as np

def test_build_counts():
    seq=[[0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 2, 2, 2, 2, 0, 0, 0]]
    counts = np.array([[8, 1, 1], [1, 3, 0], [1, 0, 3]])
    for mdl_type in [MarkovStateModel, BayesianMarkovStateModel,
                ContinuousTimeMSM]:
        mdl_instance = mdl_type()
        mdl_instance.fit(seq)
        eq(mdl_instance.countsmat_, counts)
