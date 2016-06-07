# Author: Robert McGibbon <rmcgibbo@gmail.com>
# Contributors:
# Copyright (c) 2014, Stanford University
# All rights reserved.

from __future__ import print_function, division

import numpy as np
import scipy.linalg
import scipy.optimize
from six.moves import cStringIO

from . import _ratematrix
from ._markovstatemodel import _transmat_mle_prinz
from .core import (_MappingTransformMixin, _CountsMSMMixin, _dict_compose,
                   _solve_ratemat_eigensystem, _SampleMSMMixin)
from ..base import BaseEstimator
from ..utils import printoptions


class ContinuousTimeMSM(BaseEstimator, _MappingTransformMixin,
                        _CountsMSMMixin, _SampleMSMMixin):
    """Reversible first order master equation model

    This model fits a continuous-time Markov model (master equation) from
    discrete-time integer labeled timeseries. The key estimated attribute,
    ``ratemat_``, is a matrix containing the estimated first order rate
    constants between the states. See [1] for details.

    Parameters
    ----------
    lag_time : int
        The lag time used to count the number of state to state transition
        events.
    n_timescales : int, optional
        Number of implied timescales to calculate.
    ergodic_cutoff : int, default=1
        Only the maximal strongly ergodic subgraph of the data is used to build
        an MSM. Ergodicity is determined by ensuring that each state is
        accessible from each other state via one or more paths involving edges
        with a number of observed directed counts greater than or equal to
        ``ergodic_cutoff``. Not that by setting ``ergodic_cutoff`` to 0, this
        trimming is effectively turned off.
    sliding_window : bool, default=True
        Count transitions using a window of length ``lag_time``, which is slid
        along the sequences 1 unit at a time, yielding transitions which contain
        more data but cannot be assumed to be statistically independent. Otherwise,
        the sequences are simply subsampled at an interval of ``lag_time``.
    verbose : bool, default=False
        Verbosity level
    guess : {'log', 'pseudo', array}, default='log'
        Method for determining the initial guess rate matrix, as input to the
        optimizer.

        'log':
            Initialize from matrix log of the MLE transition matrix
        'pseudo':
            Initialize from the pseudo-generator, using the 1st order taylor
            expansion of the matrix exponential.

        Otherwise, supply your own (n_states_ x n_states_) numpy array as a
        guess rate matrix.

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
    eigenvalues_ :  array of shape=(n_timescales+1)
        Largest eigenvalues of the rate matrix.
    left_eigenvectors_ : array of shape=(n_timescales+1)
        Dominant left eigenvectors of the rate matrix.
    right_eigenvectors_ : array of shape=(n_timescales+1)
        Dominant right eigenvectors of the rate matrix,

    References
    ----------
    .. [1] R. T. McGibbon and V. S. Pande, Efficient maximum likelihood
       parameterization of continuous-time Markov processes." J. Chem. Phys.
       143, 034109 (2015) http://dx.doi.org/10.1063/1.4926516

    See Also
    --------
    MarkovStateModel : discrete-time analog
    """
    def __init__(self, lag_time=1, n_timescales=None, ergodic_cutoff=1,
                 sliding_window=True, verbose=False, guess='log'):
        self.lag_time = lag_time
        self.n_timescales = n_timescales
        self.ergodic_cutoff = ergodic_cutoff
        self.verbose = verbose
        self.sliding_window = sliding_window
        self.guess = guess

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
        self.percent_retained_ = None

    def __getstate__(self):
        # gh-713
        return {k: v for k, v in self.__dict__.items()
                if k != 'optimizer_state_'}


    def fit(self, sequences, y=None):
        self._build_counts(sequences)
        return self._fit()

    def _fit(self):
        result, loglikelihoods = self._optimize()

        K = np.zeros((self.n_states_, self.n_states_))
        _ratematrix.build_ratemat(result.x, self.n_states_, K, which='K')

        self.theta_ = result.x
        self.ratemat_ = K
        self.transmat_ = scipy.linalg.expm(self.ratemat_)
        self.optimizer_state_ = result
        pi = np.exp(result.x[-self.n_states_:])
        self.populations_ = pi / pi.sum()
        self.information_ = None
        self.loglikelihoods_ = loglikelihoods

        n_timescales = self.n_timescales
        if n_timescales is None:
            n_timescales = self.n_states_ - 1
        k = n_timescales + 1
        self.eigenvalues_, self.left_eigenvectors_, self.right_eigenvectors_ = \
            _solve_ratemat_eigensystem(self.theta_, k, self.n_states_)
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
            if self.n_states_ < 20:
                print('uncertainty pi\n', self.uncertainty_pi(), file=out)
                print('uncertainty timescales\n', self.uncertainty_timescales(), file=out)

        return out.getvalue()

    def _optimize(self):
        countsmat = self.countsmat_
        n = countsmat.shape[0]
        nc2 = int(n*(n-1)/2)
        theta0 = self._initial_guess(countsmat)
        lag_time = float(self.lag_time)
        loglikelihoods = []

        options = {
            'iprint': 0 if self.verbose else -1,
            'eps': 1e-12,
            'ftol': 1e-12,
            'gtol': 1e-10,
        }

        def objective(theta):
            f, g = _ratematrix.loglikelihood(theta, countsmat, lag_time)

            if not np.isfinite(f):
                f = np.nan

            loglikelihoods.append(f)
            return -f, -g

        # this bound prevents the stationary probability for any state
        # from going below exp(-20), which helps avoid NaNs, since the
        # rate matrix involves terms like pi_i / pi_j, which get iffy
        # numerically as the populations go too close to zero. We also
        # prevent the S_ijs from being less than 0.
        bounds = [(0, None)]*nc2 + [(-20, None)]*n
        result = scipy.optimize.minimize(
            fun=objective, x0=theta0, method='L-BFGS-B', jac=True,
            bounds=bounds, options=options)

        return result, loglikelihoods

    def uncertainty_K(self):
        """Estimate of the element-wise asymptotic standard deviation
        in the rate matrix
        """
        if self.information_ is None:
            self._build_information()

        sigma_K = _ratematrix.sigma_K(
            self.information_, theta=self.theta_, n=self.n_states_)
        return sigma_K

    def uncertainty_pi(self):
        """Estimate of the element-wise asymptotic standard deviation
        in the stationary distribution.
        """
        if self.information_ is None:
            self._build_information()

        sigma_pi = _ratematrix.sigma_pi(
            self.information_, theta=self.theta_, n=self.n_states_)
        return sigma_pi

    def uncertainty_eigenvalues(self):
        """Estimate of the element-wise asymptotic standard deviation
        in the model eigenvalues
        """
        if self.information_ is None:
            self._build_information()

        sigma_eigenvalues = _ratematrix.sigma_eigenvalues(
            self.information_, theta=self.theta_, n=self.n_states_)

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
            self.information_, theta=self.theta_, n=self.n_states_)

        if self.n_timescales is None:
            return sigma_timescales
        return sigma_timescales[:self.n_timescales]

    def _initial_guess(self, countsmat):
        """Generate an initial guess for \theta.
        """

        if self.theta_ is not None:
            return self.theta_

        if self.guess == 'log':
            transmat, pi = _transmat_mle_prinz(countsmat)
            K = np.real(scipy.linalg.logm(transmat)) / self.lag_time

        elif self.guess == 'pseudo':
            transmat, pi = _transmat_mle_prinz(countsmat)
            K = (transmat - np.eye(self.n_states_)) / self.lag_time

        elif isinstance(self.guess, np.ndarray):
            pi = _solve_ratemat_eigensystem(self.guess)[1][:, 0]
            K = self.guess

        S = np.multiply(np.sqrt(np.outer(pi, 1/pi)), K)
        sflat = np.maximum(S[np.triu_indices_from(countsmat, k=1)], 0)
        theta0 = np.concatenate((sflat, np.log(pi)))
        return theta0

    def _build_information(self):
        """Build the inverse of hessian of the log likelihood at theta_
        """
        lag_time = float(self.lag_time)

        # only the "active set" of variables not at the bounds of the
        # feasible set.
        inds = np.where(self.theta_ != 0)[0]

        hessian = _ratematrix.hessian(
            self.theta_, self.countsmat_, t=lag_time, inds=inds)

        self.information_ = np.zeros((len(self.theta_), len(self.theta_)))
        self.information_[np.ix_(inds, inds)] = scipy.linalg.pinv(-hessian)

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
           of slow dynamical modes in molecular kinetics" J. Chem. Phys. 142,
           124105 (2015)
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
