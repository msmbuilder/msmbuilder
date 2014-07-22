from __future__ import print_function
import os
import numpy as np
from mdtraj.testing import eq
import scipy.sparse
from sklearn.externals.joblib import load, dump
import sklearn.pipeline
import mixtape

def test_pcca_1():
    # Make a simple dataset with four states, where there are 2 obvious macrostate basins--the remaining states interconvert quickly
    n_frames = 10000
    chunk = np.zeros(n_frames, 'int')
    rnd = lambda : np.random.randint(0, 2, n_frames)  # Generates random noise states within each basin
    # States 0 and 1 interconvert, states 2 and 3 interconvert.  
    assignments = [np.hstack((chunk + rnd(), chunk + 2  + rnd())), np.hstack((chunk + 2 + rnd(), chunk + rnd()))]

    pcca = mixtape.lumping.PCCA(2)
    macro_msm = mixtape.markovstatemodel.MarkovStateModel()

    pipeline = sklearn.pipeline.Pipeline([("pcca", pcca), ("macro_msm", macro_msm)])
    macro_assignments = pipeline.fit_transform(assignments)

    # Now let's make make the output assignments start with zero at the first position.
    i0 = macro_assignments[0][0]
    if i0 == 1:
        for m in macro_assignments:
            m *= -1
            m += 1

    eq(macro_assignments[0], np.hstack((chunk, chunk + 1)))
    eq(macro_assignments[1], np.hstack((chunk + 1, chunk)))


def test_pccaplus_1():
    # Make a simple dataset with four states, where there are 2 obvious macrostate basins--the remaining states interconvert quickly
    n_frames = 10000
    chunk = np.zeros(n_frames, 'int')
    rnd = lambda : np.random.randint(0, 2, n_frames)  # Generates random noise states within each basin
    # States 0 and 1 interconvert, states 2 and 3 interconvert.  
    assignments = [np.hstack((chunk + rnd(), chunk + 2  + rnd())), np.hstack((chunk + 2 + rnd(), chunk + rnd()))]

    pcca = mixtape.lumping.PCCAPlus(2)
    macro_msm = mixtape.markovstatemodel.MarkovStateModel()

    pipeline = sklearn.pipeline.Pipeline([("pcca", pcca), ("macro_msm", macro_msm)])
    macro_assignments = pipeline.fit_transform(assignments)

    # Now let's make make the output assignments start with zero at the first position.
    i0 = macro_assignments[0][0]
    if i0 == 1:
        for m in macro_assignments:
            m *= -1
            m += 1

    eq(macro_assignments[0], np.hstack((chunk, chunk + 1)))
    eq(macro_assignments[1], np.hstack((chunk + 1, chunk)))
