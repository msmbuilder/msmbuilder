"""Bayesian Markov state model, with MCMC sampling of the transition matrix
posterior distribution. This can be used to assess sampling uncertainty
in the MSM transition matrix and functions of the transition matrix.

TODO
----
* MCMC for models which are not constrained to be reversible is currently
  not implemented. This requires implementing ``_fit_non_reversible``, and
  a new sampler in C. This will not be very hard -- it's very close to
  the reversible version.
* Find a faster sampler. The MCMC is excruciatingly slow to converge for
  larger models.
  - Perhaps this could be improved by tweaking the proposal distribution
    (truncated normal?).
  - We can calculate the gradient of the log-posterior, so Hamiltonian
    Monte Carlo is possible. This could be a big win.
    * Note that there is a gauge-fixing issue here. When the independent
      variables are the "virtual counts", there is a scale invariance,
      which I think is a problem for the HMC, so we'd need to slightly
      reparameterize.
  - The Gibbs sampler from 10.1103/PhysRevE.82.031114 (Metzner, Weber,
    and Schutte) could be more efficient. Figure 6 of that paper seems to
    show a ~10x improvement in the mixing time vs. number of iterations,
    but each iteration takes ~10x longer in wall-clock, so it might be a
    wash.
* Implement some covergence diagonistics for the MCMC. For example, the
  Gellman-Rubin diagonstic (potential scale factor reduction). See eq.
  31 of 10.1103/PhysRevE.82.031114.
"""
from __future__ import absolute_import, division

import math
import multiprocessing
import itertools
import warnings
import numpy as np

from mixtape.utils import list_of_1d
from sklearn.base import BaseEstimator
from mixtape.markovstatemodel.core import (_MappingTransformMixin, _dict_compose,
                                           _strongly_connected_subgraph,
                                           _transition_counts,
                                           _solve_msm_eigensystem)
from mixtape.markovstatemodel._metzner_mcmc_fast import metzner_mcmc_fast
from mixtape.markovstatemodel._metzner_mcmc_slow import metzner_mcmc_slow

#-----------------------------------------------------------------------------
# Code
#-----------------------------------------------------------------------------


class BayesianMarkovStateModel(BaseEstimator, _MappingTransformMixin):
    """Bayesian Markov state model

    When ``fit()``, this model runs a Metropolis Markov chain Monte Carlo
    sampler (from Ref [1]) to estimate transition matrices. This produces
    an ensemble of ``n_samples`` transition matrices asymptotically sampled
    from the distribution :math:`P(T | C)`. This distribution gives some
    information about the statistical uncertainty in the transition matrix
    (and functions of the transition matrix), and is stored in
    ``all_transmats_``

    Parameters
    ----------
    lag_time : int
        The lag time of the model
    n_samples : int, default=100
        Total number of transition matrices to sample from the posterior
    n_steps : int, default=n_states
       Number of MCMC steps to take between sampled transition matrices. By
       default, we use ``n_steps=n_states_**2``.
    n_chains : int, default=n_procs
       Number of independent Markov chains to simulate. The requested
       number of transition matrix samples will be generated from n_chains
       independent MCMC chains.
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
    sampler : {'metzner', 'metzner_py'}
        The sampler implementation to use. 'metzer' is the sampler from Ref.
        [1] implemented in C, 'metzner_py' is a pure-python reference
        implementation.
    verbose : bool
        Enable verbose printout

    Attributes
    ----------
    n_states_ : int
        The number of states in the model
    mapping_ : dict
        Mapping between "input" labels and internal state indices used by the
        counts and transition matrix for this Markov state model. Input states
        need not necessarily be integers in (0, ..., n_states_ - 1), for
        example. The semantics of ``mapping_[i] = j`` is that state ``i`` from
        the "input space" is represented by the index ``j`` in this MSM.
    countsmat_ : array_like, shape = (n_states_, n_states_)
        Number of transition counts between states. countsmat_[i, j] is counted
        during `fit()`. The indices `i` and `j` are the "internal" indices
        described above. No correction for reversibility is made to this
        matrix.
    transmats_ : array_like, shape = (n_samples, n_states_, n_states_)
        Samples from the posterior ensemble of transition matrices.

    Notes
    -----
    Markov chain Monte Carlo can be computationally expensive. To get good
    (converged) results and acceptable performance, you'll likely need to
    play around with the ``n_samples``, ``n_steps`` and ``n_chains`` parameters.
    ``n_samples`` gives the *total* number of transition matrices sampled
    from the posterior. These samples are generated from ``n_chains`` different
    independent MCMC chains, at an interval of ``n_steps``. The total number
    of iterations of MCMC performed during ``fit()`` is ``n_samples * n_steps``.
    Increasing ``n_chains`` therefore does not alter the total number of
    iterations -- instead it controls whether those iterations occur as part
    of one long chain or multiple shorter chains (which are run in parallel
    for ``sampler=='metzner'``).

    References
    ----------
    .. [1] P. Metzner, F. Noe and C. Schutte, "Estimating the sampling error:
        Distribution of transition matrices and functions of transition
        matrices for given trajectory data." Phys. Rev. E 80 021106 (2009)
    """
    def __init__(self, lag_time=1, n_samples=100, n_steps=0, n_chains=None,
                 n_timescales=None, reversible=True, ergodic_cutoff=1,
                 prior_counts=0, random_state=None, sampler='metzner',
                 verbose=False):
        self.lag_time = lag_time
        self.n_samples = n_samples
        self.n_steps = n_steps
        self.n_chains = n_chains
        self.n_timescales = n_timescales
        self.reversible = reversible
        self.ergodic_cutoff = ergodic_cutoff
        self.prior_counts = prior_counts
        self.random_state = random_state
        self.sampler = sampler
        self.verbose = verbose

        self.mapping_ = None
        self.countsmat_ = None
        self.all_transmats_ = None
        self.n_states_ = None
        self._is_dirty = True

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
            self.all_transmats_ = fit_method(self.countsmat_)
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
        Z = countsmat + self.prior_counts
        n_steps = self.n_steps
        if n_steps == 0:
            n_steps = self.n_states_**2
        n_chains = self.n_chains
        if n_chains is None:
            n_chains = multiprocessing.cpu_count()

        # Each MCMC chain iterates for a total of chain_length steps.
        # and results are saved every n_steps. Therefore each chain
        # generates (self.n_samples / n_chains) samples. After
        # running n_chains independent chains, we get a total of
        # n_samples.
        chain_length = n_steps * int(math.ceil(self.n_samples / n_chains))

        if self.sampler == 'metzner':
            gen = metzner_mcmc_fast(
                Z, n_samples=chain_length,
                n_thin=n_steps, n_chains=n_chains,
                random_state=self.random_state)
        elif self.sampler == 'metzner_py':
            gen = itertools.chain(*(metzner_mcmc_slow(
                Z, n_samples=chain_length,
                n_thin=n_steps, random_state=self.random_state)
                                  for _ in range(n_chains)))

        else:
            raise AttributeError('sampler must be one of "metzner", "metzner_py"')

        result = np.array(list(gen))
        # For parallel 'metzner', the chains are inter-leaved in the
        # output. This can be a little confusing if you're trying to
        # look at the decorrelation time of the sampler.
        if self.sampler == 'metzner' and n_chains > 1:
            result = np.concatenate([result[i::n_chains]
                                     for i in range(n_chains)])

        # The length of result will be exactly n_samples if n_chains evenly
        # divides into the number of requested steps per chain, but otherwise
        # we round chain_length UP, so we might have generated a few extra
        # samples.
        result = result[-self.n_samples:]
        return result

    def _fit_non_reversible(self):
        raise NotImplementedError('Only the reversible sampler is currently implemented')

    def _get_eigensystem(self):
        if not self._is_dirty:
            return (self._all_eigenvalues,
                    self._all_left_eigenvectors,
                    self._all_right_eigenvectors)

        n_timescales = self.n_timescales
        if n_timescales is None:
            n_timescales = self.n_states_ - 1

        k = n_timescales + 1
        self._all_eigenvalues = []
        self._all_left_eigenvectors = []
        self._all_right_eigenvectors = []

        for transmat in self.all_transmats_:
            u, lv, rv = _solve_msm_eigensystem(transmat, k)
            self._all_eigenvalues.append(u)
            self._all_left_eigenvectors.append(lv)
            self._all_right_eigenvectors.append(rv)

        self._all_eigenvalues = np.array(self._all_eigenvalues)
        self._all_left_eigenvectors = np.array(self._all_left_eigenvectors)
        self._all_right_eigenvectors = np.array(self._all_right_eigenvectors)
        self._is_dirty = False

        return (self._all_eigenvalues,
                self._all_left_eigenvectors,
                self._all_right_eigenvectors)

    @property
    def all_timescales_(self):
        """Implied relaxation timescales each sample in the ensemble

        Returns
        -------
        timescales : array-like, shape = (n_samples, n_timescales,)
            The longest implied relaxation timescales of the each sample in
            the ensemble of transition matrices, expressed in units of
            time-step between indices in the source data supplied
            to ``fit()``.

        References
        ----------
        .. [1] Prinz, Jan-Hendrik, et al. "Markov models of molecular kinetics:
        Generation and validation." J. Chem. Phys. 134.17 (2011): 174105.
        """

        us, lvs, rvs = self._get_eigensystem()
        # make sure to leave off equilibrium distribution
        timescales = - self.lag_time / np.log(us[:, 1:])
        return timescales

    @property
    def all_eigenvalues_(self):
        """Eigenvalues of the transition matrices.

        Returns
        -------
        eigs : array-like, shape = (n_samples, n_timescales+1)
            The eigenvalues of each transition matrix in the ensemble
        """
        us, lvs, rvs = self._get_eigensystem()
        return us

    @property
    def all_left_eigenvectors_(self):
        r"""Left eigenvectors, :math:`\Phi`, of each transition matrix in the
        ensemble

        Each transition matrix's left eigenvectors are normalized such that:

          - ``lv[:, 0]`` is the equilibrium populations and is normalized
            such that `sum(lv[:, 0]) == 1``
          - The eigenvectors satisfy
            ``sum(lv[:, i] * lv[:, i] / model.populations_) == 1``.
            In math notation, this is :math:`<\phi_i, \phi_i>_{\mu^{-1}} = 1`

        Returns
        -------
        lv : array-like, shape=(n_samples, n_states, n_timescales+1)
            The columns of lv, ``lv[:, i]``, are the left eigenvectors of
            ``transmat_``.
        """
        us, lvs, rvs = self._get_eigensystem()
        return lvs

    @property
    def all_right_eigenvectors_(self):
        r"""Right eigenvectors, :math:`\Psi`, of each transition matrix in the
        ensemble

        Each transition matrix's left eigenvectors are normalized such that:

          - Weighted by the stationary distribution, the right eigenvectors
            are normalized to 1. That is,
                ``sum(rv[:, i] * rv[:, i] * self.populations_) == 1``,
            or :math:`<\psi_i, \psi_i>_{\mu} = 1`

        Returns
        -------
        rv : array-like, shape=(n_samples, n_states, n_timescales+1)
            The columns of lv, ``rv[:, i]``, are the right eigenvectors of
            ``transmat_``.
        """
        us, lvs, rvs = self._get_eigensystem()
        return rvs

    @property
    def all_populations_(self):
        us, lvs, rvs = self._get_eigensystem()
        return lvs[:, :, 0]
