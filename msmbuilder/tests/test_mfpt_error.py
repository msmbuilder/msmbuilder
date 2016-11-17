from msmbuilder.tpt import mfpt
import numpy as np


def test_create_perturb_params():
    # Test with a 10x10 counts matrix, with all entries in the counts set to 100
    countsmat = 100 * np.ones((10,10))
    params = mfpt.create_perturb_params(countsmat)
    # Check dimensions of outputs are equal to those of inputs
    for param in params:
        assert np.shape(param) == np.shape(countsmat) 


def test_perturb_tmat():
    # The transition matrix is perturbed under the CLT approximation, which is only valid for well-sampled data w.r.t. transition probability (tprob >> 1 / row-summed counts)
    countsmat = 100 * np.ones((10,10)) # 10-state MSM, 1000 counts per state, 100 transition events between states, no zero entries
    params = mfpt.create_perturb_params(countsmat)
    new_transmat = (mfpt.perturb_tmat(params[0], params[1]))
    # All transition probabilities are by design nonzero, so there should be no nonzero entries after the perturbation
    assert len(np.where(new_transmat == 0)[0]) == 0
    # Now let's assume you have a poorly sampled dataset where all elements in the counts matrix is 1
    countsmat = np.ones((10,10))
    params = mfpt.create_perturb_params(countsmat)
    new_transmat = (mfpt.perturb_tmat(params[0], params[1]))
    # Your perturbed transition matrix will have several negative values (set automatically to 0), indicating this method probably isn't appropriate for your dataset
    # (This will also cause your distribution of MFPTs to have very obvious outliers to an otherwise approximately Gaussian distribution due to the artificial zeros in the transition matrix)
    assert len(np.where(new_transmat == 0)[0] > 0)

