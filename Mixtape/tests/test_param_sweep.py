from __future__ import print_function, division
import os

import numpy as np
import numpy.testing as npt

from mixtape.msm import MarkovStateModel
from mixtape import param_sweep
from mixtape.msm import implied_timescales

def test_both():
    model = MarkovStateModel(reversible_type='mle', lag_time=1,
                             n_timescales=1) 
    # note this might break it if we ask for more than 1 timescale

    sequences = np.random.randint(20, size=(10, 1000))

    lag_times = [1, 5, 10]

    models_ref = []
    for tau in lag_times:
        msm = MarkovStateModel(reversible_type='mle', lag_time=tau,
                               n_timescales=10)
        msm.fit(sequences)

        models_ref.append(msm)

    timescales_ref = [m.timescales_ for m in models_ref]

    models = param_sweep(msm, sequences, {'lag_time' : lag_times},
                         n_jobs=2)

    timescales = implied_timescales(sequences, lag_times, msm=msm,
                                    n_timescales=10, n_jobs=2)

    print(timescales)
    print(timescales_ref)

    if np.abs(models[0].transmat_ - models[1].transmat_).sum() < 1E-6:
        raise Exception("you wrote a bad test.")

    for i in xrange(len(lag_times)):
        models[i].lag_time = lag_times[i]

        npt.assert_array_almost_equal(models[i].transmat_, models_ref[i].transmat_)
        npt.assert_array_almost_equal(timescales_ref[i], timescales[i])
