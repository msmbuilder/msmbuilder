# Author: Robert McGibbon <rmcgibbo@gmail.com>
# Contributors:
# Copyright (c) 2014, Stanford University
# All rights reserved.

from __future__ import print_function, division
import time
import numpy as np
import scipy.linalg
import scipy.optimize
from six.moves import cStringIO

from ..base import BaseEstimator
from ..utils import list_of_1d, printoptions, experimental
from . import _ratematrix
from ._markovstatemodel import _transmat_mle_prinz
from .core import (_MappingTransformMixin, _transition_counts, _dict_compose,
                   _normalize_eigensystem, _solve_ratemat_eigensystem,
                   _strongly_connected_subgraph)


class ContinuousTimeMSM(BaseEstimator, _MappingTransformMixin):
    """Reversible first order master equation model

    This model fits a reversible continuous-time Markov model for labeled
    sequence data.

    .. warning::

        This model is currently (as of December 2, 2014) experimental, and may
        undergo significant changes or bugfixes in upcoming releases.

    Parameters
    ----------
    lag_time : int
        The lag time used to count the number of state to state transition
        events.
    prior_counts : float, optional
        Add a number of "pseudo counts" to each entry in the counts matrix.
        When prior_counts == 0 (default), the assigned transition
        probability between two states with no observed transitions will be
        zero, whereas when prior_counts > 0, even this unobserved transitions
        will be given nonzero probability.
    n_timescales : int, optional
        Number of implied timescales to calculate.
    ergodic_cutoff : int, default=1
        Only the maximal strongly ergodic subgraph of the data is used to build
        an MSM. Ergodicity is determined by ensuring that each state is
        accessible from each other state via one or more paths involving edges
        with a number of observed directed counts greater than or equal to
        ``ergodic_cutoff``. Not that by setting ``ergodic_cutoff`` to 0, this
        trimming is effectively turned off.
    use_sparse : bool, default=True
        Attempt to find a sparse rate matrix.
    ftol : float, default=1e-10
        Iteration stops when the relative increase in the log-likelihood is less
        than this cutoff. Changing this cutoff can trade off between solution
        quality and runtime. For a 'quick' solution try ~1e-6, and for a vey
        high precision, go to ~1e-12 or so.
    sliding_window : bool, default=True
        Count transitions using a window of length ``lag_time``, which is slid
        along the sequences 1 unit at a time, yielding transitions which contain
        more data but cannot be assumed to be statistically independent. Otherwise,
        the sequences are simply subsampled at an interval of ``lag_time``.
    guess_ratemat : array of shape=(n_states_, n_states), optional
        Guess for the rate matrix. This can be used to warm-start the optimizer.
        Sometimes the optimizer is poorly behaved when the lag time is large,
        so it can be helpful to seed it from a model estimated using a shorter
        lag time.
    verbose : bool, default=False
        Verbosity level

    Attributes
    ----------
    n_states_ : int
        The number of states
    ratemat_ : np.ndarray, shape=(n_states_, n_state_)
        The estimated state-to-state transition rates.
    transmat_ : np.ndarray, shape=(n_states_, n_state_)
        The estimated state-to-state transition probabilities over an interval
        of 1 time unit.
    timescales_ : array of shape=(n_timescales,)
        Estimated relaxation timescales of the model.
    populations_ : np.ndarray, shape=(n_states_,)
        Estimated stationary probability distribution over the states.
    countsmat_ : array_like, shape = (n_states_, n_states_)
        Number of transition counts between states, at a time delay of ``lag_time``
        countsmat_[i, j] is counted during `fit()`.
    optimizer_state_ : object
        Contains information about the optimization termination.
    mapping_ : dict
        Mapping between "input" labels and internal state indices used by the
        counts and transition matrix for this Markov state model. Input states
        need not necessarily be integers in (0, ..., n_states_ - 1), for
        example. The semantics of ``mapping_[i] = j`` is that state ``i`` from
        the "input space" is represented by the index ``j`` in this MSM.
    theta_ : array of shape n*(n+1)/2 or shorter
        Optimized set of parameters for the model.
    information_ : np.ndarray, shape=(len(theta_), len(theta_))
        Approximate inverse of the hessian of the model log-likelihood
        evaluated at ``theta_``.
    inds_ : array of shape n*(n+1)/2 or shorter, or None
        For sparse parameterization, the indices of the non-zero elements of
        \theta.
    eigenvalues_ :  array of shape=(n_timescales+1)
        Largest eigenvalues of the rate matrix.
    left_eigenvectors_ : array of shape=(n_timescales+1)
        Dominant left eigenvectors of the rate matrix.
    right_eigenvectors_ : array of shape=(n_timescales+1)
        Dominant right eigenvectors of the rate matrix,

    See Also
    --------
    MarkovStateModel : discrete-time analog
    """
    def __init__(self, lag_time=1, prior_counts=0, n_timescales=None,
                 ergodic_cutoff=1, use_sparse=True, ftol=1e-10, sliding_window=True,
                 guess_ratemat=None, verbose=False):
        self.lag_time = lag_time
        self.prior_counts = prior_counts
        self.n_timescales = n_timescales
        self.ergodic_cutoff = ergodic_cutoff
        self.verbose = verbose
        self.use_sparse = use_sparse
        self.ftol = ftol
        self.sliding_window = sliding_window
        self.guess_ratemat = guess_ratemat

        self.inds_ = None
        self.theta_ = None
        self.ratemat_ = None
        self.transmat_ = None
        self.countsmat_ = None
        self.n_states_ = None
        self.optimizer_state_ = None
        self.mapping_ = None
        self.populations_ = None
        self.information_ = None
        self.loglikelihoods_ = None
        self.timescales_ = None
        self.eigenvalues_ = None
        self.left_eigenvectors_ = None
        self.right_eigenvectors_ = None

    @experimental('ContinuousTimeMSM')
    def fit(self, sequences, y=None):
        sequences = list_of_1d(sequences)
        lag_time = int(self.lag_time)
        if lag_time < 1:
            raise ValueError('lag_time must be >= 1')
        raw_counts, mapping = _transition_counts(
            sequences, int(lag_time), self.sliding_window)

        if self.ergodic_cutoff >= 1:
            # step 2. restrict the counts to the maximal strongly ergodic
            # subgraph
            countsmat, mapping2 = _strongly_connected_subgraph(
                lag_time * raw_counts, self.ergodic_cutoff, self.verbose)
            mapping = _dict_compose(mapping, mapping2)
        else:
            # no ergodic trimming.
            countsmat = raw_counts

        n_states = countsmat.shape[0]
        result, inds = self._optimize(countsmat + self.prior_counts)

        exptheta = np.exp(result.x)
        K = np.zeros((n_states, n_states))
        _ratematrix.build_ratemat(exptheta, n_states, inds, K, which='K')

        self.inds_ = inds
        self.theta_ = result.x
        self.ratemat_ = K
        self.transmat_ = scipy.linalg.expm(self.ratemat_)
        self.countsmat_ = countsmat
        self.n_states_ = n_states
        self.optimizer_state_ = result
        self.mapping_ = mapping
        self.populations_ = exptheta[-n_states:] / exptheta[-n_states:].sum()
        self.information_ = None

        n_timescales = self.n_timescales
        if n_timescales is None:
            n_timescales = self.n_states_ - 1
        k = n_timescales + 1
        self.eigenvalues_, self.left_eigenvectors_, self.right_eigenvectors_ = \
            _solve_ratemat_eigensystem(self.theta_, k, self.n_states_, self.inds_)
        self.timescales_ = -1 / self.eigenvalues_[1:]

        return self

    def summarize(self):
        out = cStringIO()
        with printoptions(precision=4):
            print('n_states: %s' % self.n_states_, file=out)
            print(self.optimizer_state_.message, file=out)
            print('ratemat\n', self.ratemat_, file=out)
            print('transmat\n', self.transmat_, file=out)
            print('populations\n', self.populations_, file=out)
            print('timescales\n', self.timescales_, file=out)
            print('uncertainty pi\n', self.uncertainty_pi(), file=out)
            print('uncertainty timescales\n', self.uncertainty_timescales(), file=out)

        return out.getvalue()

    def _optimize(self, countsmat):
        n = countsmat.shape[0]
        nc2 = int(n*(n-1)/2)
        theta_cutoff = np.log(1e-8)
        loglikelihoods = []

        theta0 = self._initial_guess(countsmat)
        lag_time = float(self.lag_time)

        options = {
            'iprint': 0 if self.verbose else -1,
            'ftol': self.ftol,
            #'gtol': 1e-10
        }

        def objective(theta, inds):
            start = time.time()
            f, g = _ratematrix.loglikelihood(
                theta, countsmat, n, inds, lag_time)
            print(f)
            loglikelihoods.append((f, start, len(theta)))
            return -f, -g

        # this bound prevents the stationary probability for any state
        # from going below exp(-20), which helps avoid NaNs, since the
        # rate matrix involves terms like pi_i / pi_j, which get iffy
        # numerically as the populations go too close to zero. We also
        # prevent the S_ijs from getting similarly small, since in the next
        # optimization step using the sparse parameterizetion, they can get
        # truncated.
        bounds0 = [(-20, None)]*nc2 + [(-20, 0)]*n
        inds0 = None
        options0 = dict(options, maxiter=max(n//10, 25)) if self.use_sparse else options
        result0 = scipy.optimize.minimize(
            fun=objective, x0=theta0, method='L-BFGS-B', jac=True,
            bounds=bounds0, args=(inds0,), options=options0)

        # now, try rerunning the optimization with theta restricted to only
        # the dominant elements -- try zeroing out the elements that are too
        # small.
        inds1 = np.concatenate((
            np.where(result0.x[:nc2] > theta_cutoff)[0], nc2 + np.arange(n)))

        if (len(inds1) == nc2 + n) or (not self.use_sparse):
            value = (result0, inds0)
        else:
            bounds1 = [bounds0[i] for i in inds1]
            result1 = scipy.optimize.minimize(
                fun=objective, x0=result0.x[inds1], method='L-BFGS-B', jac=True,
                bounds=bounds1, args=(inds1,), options=options)

            if result1.fun < result0.fun:
                if self.verbose:
                    print('[ContinuousTimeMSM] %d rates pegged to zero' %
                          (nc2 + n - len(inds1)))
                value = (result1, inds1)
            else:
                if self.verbose:
                    print('[ContinuousTimeMSM] No rates pegged to zero')
                value = (result0, inds0)

        self.loglikelihoods_ = np.array(loglikelihoods)
        return value

    def uncertainty_K(self):
        """Estimate of the element-wise asymptotic standard deviation
        in the rate matrix
        """
        if self.information_ is None:
            self._build_information()

        sigma_K = _ratematrix.sigma_K(
            self.information_, theta=self.theta_, n=self.n_states_,
            inds=self.inds_)
        return sigma_K

    def uncertainty_pi(self):
        """Estimate of the element-wise asymptotic standard deviation
        in the stationary distribution.
        """
        if self.information_ is None:
            self._build_information()

        sigma_pi = _ratematrix.sigma_pi(
            self.information_, theta=self.theta_, n=self.n_states_,
            inds=self.inds_)
        return sigma_pi

    def uncertainty_eigenvalues(self):
        """Estimate of the element-wise asymptotic standard deviation
        in the model eigenvalues
        """
        if self.information_ is None:
            self._build_information()

        sigma_eigenvalues = _ratematrix.sigma_eigenvalues(
            self.information_, theta=self.theta_, n=self.n_states_,
            inds=self.inds_)

        if self.n_timescales is None:
            return sigma_eigenvalues
        return np.nan_to_num(sigma_eigenvalues[:self.n_timescales+1])

    def uncertainty_timescales(self):
        """Estimate of the element-wise asymptotic standard deviation
        in the model relaxation timescales.
        """
        if self.information_ is None:
            self._build_information()

        sigma_timescales = _ratematrix.sigma_timescales(
            self.information_, theta=self.theta_, n=self.n_states_,
            inds=self.inds_)
        if self.n_timescales is None:
            return sigma_timescales
        return sigma_timescales[:self.n_timescales]

    def _initial_guess(self, countsmat):
        """Generate an initial guess for \theta.
        """
        transmat, pi = _transmat_mle_prinz(countsmat + self.prior_counts)
        K = _ratematrix.logm(transmat, pi, self.lag_time)
        K = np.maximum(K, 1e-10)
        S = np.multiply(np.sqrt(np.outer(pi, 1/pi)), K)

        sflat = np.maximum(S[np.triu_indices_from(countsmat, k=1)], 1e-10)
        theta0 = np.concatenate((np.maximum(-19, np.log(sflat)), np.log(pi)))
        return theta0

    def _build_information(self):
        """Build the inverse of hessian of the log likelihood at theta_
        """
        lag_time = float(self.lag_time)

        hessian = _ratematrix.hessian(
            self.theta_, self.countsmat_, self.n_states_, inds=self.inds_,
            t=lag_time)

        self.information_ = scipy.linalg.pinv(-hessian)

    @property
    def score_(self):
        """Training score of the model, computed as the generalized matrix,
        Rayleigh quotient, the sum of the first `n_components` eigenvalues
        """
        return np.exp(self.eigenvalues_).sum()

    def score(self, sequences, y=None):
        """Score the model on new data using the generalized matrix Rayleigh
        quotient

        Parameters
        ----------
        sequences : list of array-like
            List of sequences, or a single sequence. Each sequence should be a
            1D iterable of state labels. Labels can be integers, strings, or
            other orderable objects.

        Returns
        -------
        gmrq : float
            Generalized matrix Rayleigh quotient. This number indicates how
            well the top ``n_timescales+1`` eigenvectors of this model perform
            as slowly decorrelating collective variables for the new data in
            ``sequences``.

        References
        ----------
        .. [1] McGibbon, R. T. and V. S. Pande, "Variational cross-validation
           of slow dynamical modes in molecular kinetics"
           http://arxiv.org/abs/1407.8083 (2014)
        """
        # eigenvectors from the model we're scoring, `self`
        V = self.right_eigenvectors_

        m2 = self.__class__(**self.get_params())
        m2.fit(sequences)

        if self.mapping_ != m2.mapping_:
            V = self._map_eigenvectors(V, m2.mapping_)

        S = np.diag(m2.populations_)
        C = S.dot(m2.transmat_)

        try:
            trace = np.trace(V.T.dot(C.dot(V)).dot(np.linalg.inv(V.T.dot(S.dot(V)))))
        except np.linalg.LinAlgError:
            trace = np.nan

        return trace

    def _map_eigenvectors(self, V, other_mapping):
        self_inverse_mapping = {v: k for k, v in self.mapping_.items()}
        transform_mapping = _dict_compose(self_inverse_mapping, other_mapping)
        source_indices, dest_indices = zip(*transform_mapping.items())

        mapped_V = np.zeros((len(other_mapping), V.shape[1]))
        mapped_V[dest_indices, :] = np.take(V, source_indices, axis=0)
        return mapped_V

    def _solve_eigensystem(self):
        n = self.n_states_

        n_timescales = self.n_timescales
        if n_timescales is None:
            n_timescales = self.n_states_ - 1
        k = n_timescales + 1

        S = np.zeros((n, n))
        exptheta = np.exp(self.theta_)
        _ratematrix.build_ratemat(exptheta, n, self.inds_, S, which='S')
        u, lv, rv = map(np.asarray, _ratematrix.eig_K(S, n, exptheta[-n:], 'S'))
        order = np.argsort(-u)
        u = u[order[:k]]
        lv = lv[:, order[:k]]
        rv = rv[:, order[:k]]

        return _normalize_eigensystem(u, lv, rv)
