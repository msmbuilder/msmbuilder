from __future__ import print_function

import numpy as np
from sklearn.pipeline import Pipeline

from msmbuilder.lumping import PCCA, PCCAPlus
from msmbuilder.msm import MarkovStateModel


def _metastable_system():
    # Make a simple dataset with four states, where there are 2 obvious
    # macrostate. States 0 and 1 interconvert, states 2 and 3 interconvert.
    n = 250
    random = np.random.RandomState(1)
    assignments = np.concatenate([
        0 + random.randint(2, size=n),  # states 0, 1 interconverting
        2 + random.randint(2, size=n),  # states 2, 3 intercoverting
        0 + random.randint(2, size=n),  # states 0, 1 interconverting
        2 + random.randint(2, size=n)  # states 2, 3 interconverting
    ])

    # the true (2-state) macrostate assignments
    macro_assignments = np.concatenate([
        np.zeros(n, dtype=int),
        np.ones(n, dtype=int),
        np.zeros(n, dtype=int),
        np.ones(n, dtype=int)
    ])

    return assignments, macro_assignments


def test_pcca_1():
    assignments, ref_macrostate_assignments = _metastable_system()
    pipeline = Pipeline([
        ('msm', MarkovStateModel()),
        ('pcca+', PCCA(2))
    ])
    macro_assignments = pipeline.fit_transform(assignments)[0]

    # we need to consider any permutation of the state labels when we
    # test for equality. Since it's only a 2-state that's simple using
    # the logical_not to flip the assignments.
    opposite = np.logical_not(ref_macrostate_assignments)
    assert (np.all(macro_assignments == ref_macrostate_assignments) or
            np.all(macro_assignments == opposite))


def test_pcca_plus_1():
    assignments, ref_macrostate_assignments = _metastable_system()
    pipeline = Pipeline([
        ('msm', MarkovStateModel()),
        ('pcca+', PCCAPlus(2))
    ])
    macro_assignments = pipeline.fit_transform(assignments)[0]
    # we need to consider any permutation of the state labels when we
    # test for equality. Since it's only a 2-state that's simple using
    # the logical_not to flip the assignments.
    opposite = np.logical_not(ref_macrostate_assignments)
    assert (np.all(macro_assignments == ref_macrostate_assignments) or
            np.all(macro_assignments == opposite))


def test_from_msm():
    assignments, _ = _metastable_system()
    msm = MarkovStateModel()
    msm.fit(assignments)
    pcca = PCCA.from_msm(msm, 2)

    msm = MarkovStateModel()
    msm.fit(assignments)
    pccaplus = PCCAPlus.from_msm(msm, 2)


def test_ntimescales_1():
    # see issue #603
    trajs = [np.random.randint(0, 30, size=500) for _ in range(15)]

    pccap = PCCAPlus(n_macrostates=11).fit(trajs)
    lumped_trajs = pccap.transform(trajs)
    assert len(np.unique(lumped_trajs)) == 11


def test_ntimescales_2():
    # see issue #603
    trajs = [np.random.randint(0, 30, size=500) for _ in range(15)]
    msm = MarkovStateModel().fit(trajs)

    pccap = PCCAPlus.from_msm(msm, 11)
    lumped_trajs = pccap.transform(trajs)
    assert len(np.unique(lumped_trajs)) == 11


def test_ntimescales_3():
    # see issue #603
    trajs = [np.random.randint(0, 30, size=500) for _ in range(15)]
    msm = MarkovStateModel(n_timescales=10).fit(trajs)

    pccap = PCCAPlus.from_msm(msm, 11)
    lumped_trajs = pccap.transform(trajs)
    assert len(np.unique(lumped_trajs)) == 11
