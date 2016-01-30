from __future__ import print_function, division

import numpy as np
import numpy.testing as npt

from msmbuilder.msm import MarkovStateModel
from msmbuilder.msm import implied_timescales
from msmbuilder.utils import param_sweep


def test_both():
    sequences = [np.random.randint(20, size=1000) for _ in range(10)]
    lag_times = [1, 5, 10]

    models_ref = []
    for tau in lag_times:
        msm = MarkovStateModel(reversible_type='mle', lag_time=tau,
                               n_timescales=10)
        msm.fit(sequences)
        models_ref.append(msm)

    timescales_ref = [m.timescales_ for m in models_ref]

    model = MarkovStateModel(reversible_type='mle', lag_time=1, n_timescales=10)
    models = param_sweep(model, sequences, {'lag_time': lag_times}, n_jobs=2)
    timescales = implied_timescales(sequences, lag_times, msm=model,
                                    n_timescales=10, n_jobs=2)

    print(timescales)
    print(timescales_ref)

    if np.abs(models[0].transmat_ - models[1].transmat_).sum() < 1E-6:
        raise Exception("you wrote a bad test.")

    for i in range(len(lag_times)):
        npt.assert_array_almost_equal(models[i].transmat_,
                                      models_ref[i].transmat_)
        npt.assert_array_almost_equal(timescales_ref[i], timescales[i])


def test_multi_params():
    msm = MarkovStateModel()
    param_grid = {
        'lag_time': [1, 2, 3],
        'reversible_type': ['mle', 'transpose']
    }

    sequences = np.random.randint(20, size=(10, 1000))
    models = param_sweep(msm, sequences, param_grid, n_jobs=2)
    assert len(models) == 6

    # I don't know what the order should be, so I'm just going
    # to check that there are no duplicates
    params = []
    for m in models:
        params.append('%s%d' % (m.reversible_type, m.lag_time))

    for l in param_grid['lag_time']:
        for s in param_grid['reversible_type']:
            assert ('%s%d' % (s, l)) in params

    # this is redundant, but w/e
    assert len(set(params)) == 6


def test_ntimescales():
    # see issue #603
    trajs = [np.random.randint(0, 30, 500) for _ in range(5)]
    its = implied_timescales(trajs, [1, 2, 3], n_timescales=11)
    assert its.shape[1] == 11
