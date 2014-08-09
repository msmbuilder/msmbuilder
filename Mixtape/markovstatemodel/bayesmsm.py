from __future__ import absolute_import

import warnings
import numpy as np
import scipy.linalg

from mdtraj.utils import ensure_type
from mixtape.utils import list_of_1d
from sklearn.utils import check_random_state
from sklearn.base import BaseEstimator
from .core import (_MappingTransformMixin, _dict_compose,
                   _strongly_connected_subgraph,
                   _transition_counts,
                   ndgrid_msm_likelihood_score)

#-----------------------------------------------------------------------------
# Code
#-----------------------------------------------------------------------------


class BayesianMarkovStateModel(BaseEstimator, _MappingTransformMixin):
    """Bayesian Markov state model

    Parameters
    ----------
    lag_time : int
        The lag time of the model
    n_burn : int, default=0
    n_samples : int, default=100
    n_thin : int, default=1
    n_timescales : int, optional
        The number of dynamical timescales to calculate when diagonalizing
        the transition matrix.
    reversible : bool, default=True
         Enforce reversibility during transition matrix sampling
    ergodic_cutoff : int, default=1
        Only the maximal strongly ergodic subgraph of the data is used to build
        an MSM. Ergodicity is determined by ensuring that each state is
        accessible from each other state via one or more paths involving edges
        with a number of observed directed counts greater than or equal to
        ``ergodic_cutoff``. Not that by setting ``ergodic_cutoff`` to 0, this
        trimming is effectively turned off.
    prior_counts : float, optional
        Add a number of "pseudo counts" to each entry in the counts matrix.
        When prior_counts == 0 (default), the assigned transition
        probability between two states with no observed transitions will be
        zero, whereas when prior_counts > 0, even this unobserved transitions
        will be given nonzero probability.
    random_state : int or RandomState instance or None (default)
        Pseudo Random Number generator seed control. If None, use the
        numpy.random singleton.
    verbose : bool
        Enable verbose printout

    Attributes
    ----------


    References
    ----------
    .. [1] P. Metzner, F. Noe and C. Schutte, "Estimating the sampling error:
        Distribution of transition matrices and functions of transition
        matrices for given trajectory data." Phys. Rev. E 80 021106 (2009)
    """
    def __init__(self, lag_time, n_burn=0, n_samples=100, n_thin=1,
                 n_timescales=None, reversible=True, ergodic_cutoff=1,
                 prior_counts=0, random_state=None, verbose=False):
        self.lag_time = lag_time
        self.n_burn = n_burn
        self.n_samples = n_samples
        self.n_thin = n_thin
        self.n_timescales = n_timescales
        self.reversible = reversible
        self.ergodic_cutoff = ergodic_cutoff
        self.prior_counts = prior_counts
        self.random_state = random_state
        self.verbose = verbose

        self.mapping_ = None
        self.countsmat_ = None
        self.transmat_samples_ = None
        self.n_states_ = None

    def fit(self, sequences, y=None):
        sequences = list_of_1d(sequences)
        raw_counts, mapping = _transition_counts(sequences, self.lag_time)

        if self.ergodic_cutoff >= 1:
            self.countsmat_, mapping2 = _strongly_connected_subgraph(
                raw_counts, self.ergodic_cutoff, self.verbose)
            self.mapping_ = _dict_compose(mapping, mapping2)
        else:
            self.countsmat_ = raw_counts
            self.mapping_ = mapping

        self.n_states_ = self.countsmat_.shape[0]
        fit_method_map = {
            True: self._fit_reversible,
            False: self._fit_non_reversible}

        try:
            fit_method = fit_method_map[self.reversible]
            self.transmat_samples_ = fit_method(self.countsmat_)
        except KeyError:
            raise ValueError('reversible_type must be one of %s: %s' % (
                ', '.join(fit_method_map.keys()), self.reversible_type))

        return self

    def _fit_reversible(self, countsmat):
        if self.ergodic_cutoff < 1:
            with warnings.catch_warnings(record=True):
                warnings.simplefilter("always")
                warnings.warn("reversible=True and ergodic_cutoff < 1 "
                              "are not generally compatible")
        pass

    def _fit_non_reversible(self):
        raise NotImplementedError()
        