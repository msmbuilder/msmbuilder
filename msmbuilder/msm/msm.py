# Author: Robert McGibbon <rmcgibbo@gmail.com>
# Contributors:
# Copyright (c) 2014, Stanford University
# All rights reserved.

#-----------------------------------------------------------------------------
# Imports
#-----------------------------------------------------------------------------

from __future__ import print_function, division, absolute_import

import sys
import warnings
import operator
import numpy as np
import scipy.linalg

from sklearn.utils import check_random_state
from ..utils import list_of_1d
from ..base import BaseEstimator
from ._markovstatemodel import _transmat_mle_prinz
from .core import (_MappingTransformMixin, _CountsMSMMixin,
                   _dict_compose,
                   _transition_counts,
                   _solve_msm_eigensystem, _SampleMSMMixin)

__all__ = ['MarkovStateModel']

#-----------------------------------------------------------------------------
# Code
#-----------------------------------------------------------------------------

class MarkovStateModel(BaseEstimator, _MappingTransformMixin,
                        _SampleMSMMixin, _CountsMSMMixin):
    """Reversible Markov State Model

    This model fits a first-order Markov model to a dataset of integer-valued
    timeseries. The key estimated attribute, ``transmat_`` is a matrix
    containing the estimated probability of transitioning between pairs
    of states in the duration specified by ``lag_time``.

    Unless otherwise specified, the model is constrained to be reversible
    (satisfy detailed balance), which is appropriate for equilibrium chemical
    systems.

    Parameters
    ----------
    lag_time : int
        The lag time of the model
    n_timescales : int, optional
        The number of dynamical timescales to calculate when diagonalizing
        the transition matrix. If not specified, it will compute n_states - 1
    reversible_type : {'mle', 'transpose', None}
        Method by which the reversibility of the transition matrix
        is enforced. 'mle' uses a maximum likelihood method that is
        solved by numerical optimization, and 'transpose'
        uses a more restrictive (but less computationally complex)
        direct symmetrization of the expected number of counts.
    ergodic_cutoff : float or {'on', 'off'}, default='on'
        Only the maximal strongly ergodic subgraph of the data is used to build
        an MSM. Ergodicity is determined by ensuring that each state is
        accessible from each other state via one or more paths involving edges
        with a number of observed directed counts greater than or equal to
        ``ergodic_cutoff``. By setting ``ergodic_cutoff`` to 0 or
        'off', this trimming is turned off. Setting it to 'on' sets the
        cutoff to the minimal possible count value.
    prior_counts : float, optional
        Add a number of "pseudo counts" to each entry in the counts matrix
        after ergodic trimming.  When prior_counts == 0 (default), the assigned
        transition probability between two states with no observed transitions
        will be zero, whereas when prior_counts > 0, even this unobserved
        transitions will be given nonzero probability.
    sliding_window : bool, optional
        Count transitions using a window of length ``lag_time``, which is slid
        along the sequences 1 unit at a time, yielding transitions which
        contain more data but cannot be assumed to be statistically
        independent. Otherwise, the sequences are simply subsampled at an
        interval of ``lag_time``.
    verbose : bool
        Enable verbose printout

    References
    ----------
    .. [1] Prinz, Jan-Hendrik, et al. "Markov models of molecular kinetics:
       Generation and validation." J Chem. Phys. 134.17 (2011): 174105.
    .. [2] Pande, V. S., K. A. Beauchamp, and G. R. Bowman. "Everything you
       wanted to know about Markov State Models but were afraid to ask"
       Methods 52.1 (2010): 99-105.

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
    transmat_ : array_like, shape = (n_states_, n_states_)
        Maximum likelihood estimate of the reversible transition matrix.
        The indices `i` and `j` are the "internal" indices described above.
    populations_ : array, shape = (n_states_,)
        The equilibrium population (stationary eigenvector) of transmat_
    """

    def __init__(self, lag_time=1, n_timescales=None, reversible_type='mle',
                 ergodic_cutoff='on', prior_counts=0, sliding_window=True,
                 verbose=True):
        self.reversible_type = reversible_type
        self.lag_time = lag_time
        self.n_timescales = n_timescales
        self.prior_counts = prior_counts
        self.sliding_window = sliding_window
        self.verbose = verbose
        self.ergodic_cutoff = ergodic_cutoff

        # Keep track of whether to recalculate eigensystem
        self._is_dirty = True
        # Cached eigensystem
        self._eigenvalues = None
        self._left_eigenvectors = None
        self._right_eigenvectors = None

        self.mapping_ = None
        self.countsmat_ = None
        self.transmat_ = None
        self.n_states_ = None
        self.populations_ = None
        self.percent_retained_ = None


    def fit(self, sequences, y=None):
        """Estimate model parameters.

        Parameters
        ----------
        sequences : list of array-like
            List of sequences, or a single sequence. Each sequence should be a
            1D iterable of state labels. Labels can be integers, strings, or
            other orderable objects.

        Returns
        -------
        self

        Notes
        -----
        `None` and `NaN` are recognized immediately as invalid labels.
        Therefore, transition counts from or to a sequence item which is NaN or
        None will not be counted. The mapping_ attribute will not include the
        NaN or None.
        """
        self._build_counts(sequences)

        # use a dict like a switch statement: dispatch to different
        # transition matrix estimators depending on the value of
        # self.reversible_type
        fit_method_map = {
            'mle': self._fit_mle,
            'transpose': self._fit_transpose,
            'none': self._fit_asymetric}

        try:
            # pull out the appropriate method
            fit_method = fit_method_map[str(self.reversible_type).lower()]
            # step 3. estimate transition matrix
            self.transmat_, self.populations_ = fit_method(self.countsmat_)
        except KeyError:
            raise ValueError('reversible_type must be one of %s: %s' % (
                ', '.join(fit_method_map.keys()), self.reversible_type))

        self._is_dirty = True
        return self

    def _fit_mle(self, counts):
        if self._parse_ergodic_cutoff() <= 0 and self.prior_counts == 0:
            warnings.warn("reversible_type='mle' and ergodic_cutoff <= 0 "
                          "are not generally compatible")

        transmat, populations = _transmat_mle_prinz(
            counts + self.prior_counts)
        return transmat, populations

    def _fit_transpose(self, counts):
        rev_counts = 0.5 * (counts + counts.T) + self.prior_counts

        populations = rev_counts.sum(axis=0)
        populations /= populations.sum(dtype=float)
        transmat = rev_counts.astype(float) / rev_counts.sum(axis=1)[:, None]
        return transmat, populations

    def _fit_asymetric(self, counts):
        rc = counts + self.prior_counts
        transmat = rc.astype(float) / rc.sum(axis=1)[:, None]

        u, lv = scipy.linalg.eig(transmat, left=True, right=False)
        order = np.argsort(-np.real(u))
        u = np.real_if_close(u[order])
        lv = np.real_if_close(lv[:, order])

        populations = lv[:, 0]
        populations /= populations.sum(dtype=float)

        return transmat, populations

    def eigtransform(self, sequences, right=True, mode='clip'):
        r"""Transform a list of sequences by projecting the sequences onto
        the first `n_timescales` dynamical eigenvectors.

        Parameters
        ----------
        sequences : list of array-like
            List of sequences, or a single sequence. Each sequence should be a
            1D iterable of state labels. Labels can be integers, strings, or
            other orderable objects.

        right : bool
            Which eigenvectors to map onto. Both the left (:math:`\Phi`) and
            the right (:math`\Psi`) eigenvectors of the transition matrix are
            commonly used, and differ in their normalization. The two sets of
            eigenvectors are related by the stationary distribution ::

                \Phi_i(x) = \Psi_i(x) * \mu(x)

            In the MSM literature, the right vectors (default here) are
            approximations to the transfer operator eigenfunctions, whereas
            the left eigenfunction are approximations to the propagator
            eigenfunctions. For more details, refer to reference [1].

        mode : {'clip', 'fill'}
            Method by which to treat labels in `sequences` which do not have
            a corresponding index. This can be due, for example, to the ergodic
            trimming step.

           ``clip``
               Unmapped labels are removed during transform. If they occur
               at the beginning or end of a sequence, the resulting transformed
               sequence will be shorted. If they occur in the middle of a
               sequence, that sequence will be broken into two (or more)
               sequences. (Default)
            ``fill``
               Unmapped labels will be replaced with NaN, to signal missing
               data. [The use of NaN to signal missing data is not fantastic,
               but it's consistent with current behavior of the ``pandas``
               library.]

        Returns
        -------
        transformed : list of 2d arrays
            Each element of transformed is an array of shape ``(n_samples,
            n_timescales)`` containing the transformed data.

        References
        ----------
        .. [1] Prinz, Jan-Hendrik, et al. "Markov models of molecular kinetics:
           Generation and validation." J. Chem. Phys. 134.17 (2011): 174105.
        """

        result = []
        for y in self.transform(sequences, mode=mode):
            if right:
                op = self.right_eigenvectors_[:, 1:]
            else:
                op = self.left_eigenvectors_[:, 1:]

            is_finite = np.isfinite(y)
            if not np.all(is_finite):
                value = np.empty((y.shape[0], op.shape[1]))
                value[is_finite, :] = np.take(op, y[is_finite].astype(np.int), axis=0)
                value[~is_finite, :] = np.nan
            else:
                value = np.take(op, y, axis=0)
            result.append(value)

        return result

    def sample(self, state=None, n_steps=100, random_state=None):
        warnings.warn("msm.sample() has been renamed as msm.sample_discrete() and will be removed in MSMBuilder3.4")
        return self.sample_discrete(state=state, n_steps=n_steps,
                                    random_state=random_state)

    def score_ll(self, sequences):
        r"""log of the likelihood of sequences with respect to the model

        Parameters
        ----------
        sequences : list of array-like
            List of sequences, or a single sequence. Each sequence should be a
            1D iterable of state labels. Labels can be integers, strings, or
            other orderable objects.

        Returns
        -------
        loglikelihood : float
            The natural log of the likelihood, computed as
            :math:`\sum_{ij} C_{ij} \log(P_{ij})`
            where C is a matrix of counts computed from the input sequences.
        """
        counts, mapping = _transition_counts(sequences)
        if not set(self.mapping_.keys()).issuperset(mapping.keys()):
            return -np.inf
        inverse_mapping = {v: k for k, v in mapping.items()}

        # maps indices in counts to indices in transmat
        m2 = _dict_compose(inverse_mapping, self.mapping_)
        indices = [e[1] for e in sorted(m2.items())]

        transmat_slice = self.transmat_[np.ix_(indices, indices)]
        return np.nansum(np.log(transmat_slice) * counts)

    def _get_eigensystem(self):
        if not self._is_dirty:
            return (self._eigenvalues,
                    self._left_eigenvectors,
                    self._right_eigenvectors)

        n_timescales = min(self.n_timescales if self.n_timescales is not None
                           else self.n_states_ - 1, self.n_states_ - 1)

        k = n_timescales + 1
        u, lv, rv = _solve_msm_eigensystem(self.transmat_, k)
        self._eigenvalues = u
        self._left_eigenvectors = lv
        self._right_eigenvectors = rv

        self._is_dirty = False

        return u, lv, rv

    def summarize(self):
        """Return some diagnostic summary statistics about this Markov model
        """

        doc = '''Markov state model
------------------
Lag time         : {lag_time}
Reversible type  : {reversible_type}
Ergodic cutoff   : {ergodic_cutoff}
Prior counts     : {prior_counts}

Number of states : {n_states}
Number of nonzero entries in counts matrix : {counts_nz} ({percent_counts_nz}%)
Nonzero counts matrix entries:
    Min.   : {cnz_min:.1f}
    1st Qu.: {cnz_1st:.1f}
    Median : {cnz_med:.1f}
    Mean   : {cnz_mean:.1f}
    3rd Qu.: {cnz_3rd:.1f}
    Max.   : {cnz_max:.1f}

Total transition counts :
    {cnz_sum} counts
Total transition counts / lag_time:
    {cnz_sum_per_lag} units
Timescales:
    [{ts}]  units
'''
        counts_nz = np.count_nonzero(self.countsmat_)
        cnz = self.countsmat_[np.nonzero(self.countsmat_)]

        return doc.format(
            lag_time=self.lag_time,
            reversible_type=self.reversible_type,
            ergodic_cutoff=self.ergodic_cutoff,
            prior_counts=self.prior_counts,
            n_states=self.n_states_,
            counts_nz=counts_nz,
            percent_counts_nz=(100 * counts_nz / self.countsmat_.size),
            cnz_min=np.min(cnz),
            cnz_1st=np.percentile(cnz, 25),
            cnz_med=np.percentile(cnz, 50),
            cnz_mean=np.mean(cnz),
            cnz_3rd=np.percentile(cnz, 75),
            cnz_max=np.max(cnz),
            cnz_sum=np.sum(cnz),
            cnz_sum_per_lag=np.sum(cnz)/self.lag_time,
            ts=', '.join(['{:.2f}'.format(t) for t in self.timescales_]),
            )

    @property
    def score_(self):
        """Training score of the model, computed as the generalized matrix,
        Rayleigh quotient, the sum of the first `n_components` eigenvalues
        """
        return self.eigenvalues_.sum()


    def score(self, sequences, y=None):
        """Score the model on new data using the generalized matrix Rayleigh quotient

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
            well the top ``n_timescales+1`` eigenvectors of this MSM perform as
            slowly decorrelating collective variables for the new data in
            ``sequences``.

        References
        ----------
        .. [1] McGibbon, R. T. and V. S. Pande, "Variational cross-validation
           of slow dynamical modes in molecular kinetics" J. Chem. Phys. 142,
           124105 (2015)
        """
        # eigenvectors from the model we're scoring, `self`
        V = self.right_eigenvectors_

        # Note: How do we deal with regularization parameters like prior_counts
        # here? I'm not sure. Should C and S be estimated using self's
        # regularization parameters?
        m2 = self.__class__(**self.get_params())
        m2.fit(sequences)

        if self.mapping_ != m2.mapping_:
            V = self._map_eigenvectors(V, m2.mapping_)
            # we need to map this model's eigenvectors
            # into the m2 space

        # How well do they diagonalize S and C, which are
        # computed from the new test data?
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

        #print(source_indices, dest_indices)
        mapped_V = np.zeros((len(other_mapping), V.shape[1]))
        mapped_V[dest_indices, :] = np.take(V, source_indices, axis=0)
        return mapped_V

    @property
    def timescales_(self):
        """Implied relaxation timescales of the model.

        The relaxation of any initial distribution towards equilibrium is
        given, according to this model, by a sum of terms -- each corresponding
        to the relaxation along a specific direction (eigenvector) in state
        space -- which decay exponentially in time. See equation 19. from [1].

        Returns
        -------
        timescales : array-like, shape = (n_timescales,)
            The longest implied relaxation timescales of the model, expressed
            in units of time-step between indices in the source data supplied
            to ``fit()``.

        References
        ----------
        .. [1] Prinz, Jan-Hendrik, et al. "Markov models of molecular kinetics:
        Generation and validation." J. Chem. Phys. 134.17 (2011): 174105.
        """
        u, lv, rv = self._get_eigensystem()

        # make sure to leave off equilibrium distribution
        with np.errstate(invalid='ignore', divide='ignore'):
            timescales = - self.lag_time / np.log(u[1:])
        return timescales

    @property
    def eigenvalues_(self):
        """Eigenvalues of the transition matrix.
        """
        u, lv, rv = self._get_eigensystem()
        return u

    @property
    def left_eigenvectors_(self):
        r"""Left eigenvectors, :math:`\Phi`, of the transition matrix.

        The left eigenvectors are normalized such that:

          - ``lv[:, 0]`` is the equilibrium populations and is normalized
            such that `sum(lv[:, 0]) == 1``
          - The eigenvectors satisfy
            ``sum(lv[:, i] * lv[:, i] / model.populations_) == 1``.
            In math notation, this is :math:`<\phi_i, \phi_i>_{\mu^{-1}} = 1`

        Returns
        -------
        lv : array-like, shape=(n_states, n_timescales+1)
            The columns of lv, ``lv[:, i]``, are the left eigenvectors of
            ``transmat_``.
        """
        u, lv, rv = self._get_eigensystem()
        return lv

    @property
    def right_eigenvectors_(self):
        r"""Right eigenvectors, :math:`\Psi`, of the transition matrix.

        The right eigenvectors are normalized such that:

          - Weighted by the stationary distribution, the right eigenvectors
            are normalized to 1. That is,
                ``sum(rv[:, i] * rv[:, i] * self.populations_) == 1``,
            or :math:`<\psi_i, \psi_i>_{\mu} = 1`

        Returns
        -------
        rv : array-like, shape=(n_states, n_timescales+1)
            The columns of lv, ``rv[:, i]``, are the right eigenvectors of
            ``transmat_``.
        """
        u, lv, rv = self._get_eigensystem()
        return rv

    @property
    def state_labels_(self):
        return [k for k, v in sorted(self.mapping_.items(),
                                     key=operator.itemgetter(1))]

    def uncertainty_eigenvalues(self):
        """Estimate of the element-wise asymptotic standard deviation
        in the model eigenvalues.

        Returns
        -------
        sigma_eigs : np.array, shape=(n_timescales+1,)
            The estimated symptotic standard deviation in the eigenvalues.

        References
        ----------
        .. [1] Hinrichs, Nina Singhal, and Vijay S. Pande. "Calculation of
           the distribution of eigenvalues and eigenvectors in Markovian state
           models for molecular dynamics." J. Chem. Phys. 126.24 (2007): 244101.
        """
        if self.reversible_type is None:
            raise NotImplementedError('reversible_type must be "mle" or "transpose"')

        n_timescales = min(self.n_timescales if self.n_timescales is not None
                           else self.n_states_ - 1, self.n_states_ - 1)

        u, lv, rv = self._get_eigensystem()

        sigma2 = np.zeros(n_timescales + 1)
        for k in range(n_timescales + 1):
            dLambda_dT = np.outer(lv[:, k], rv[:, k])
            for i in range(self.n_states_):
                ui = self.countsmat_[:, i]
                wi = np.sum(ui)
                cov = wi*np.diag(ui) - np.outer(ui, ui)
                quad_form = dLambda_dT[i].dot(cov).dot(dLambda_dT[i])
                sigma2[k] += quad_form / (wi**2*(wi+1))
        return np.sqrt(sigma2)

    def uncertainty_timescales(self):
        """Estimate of the element-wise asymptotic standard deviation
        in the model implied timescales.

        Returns
        -------
        sigma_timescales : np.array, shape=(n_timescales,)
            The estimated symptotic standard deviation in the implied
            timescales.

        References
        ----------
        .. [1] Hinrichs, Nina Singhal, and Vijay S. Pande. "Calculation of
           the distribution of eigenvalues and eigenvectors in Markovian state
           models for molecular dynamics." J. Chem. Phys. 126.24 (2007): 244101.
        """
        # drop the first eigenvalue
        u = self.eigenvalues_[1:]
        sigma_eigs = self.uncertainty_eigenvalues()[1:]

        sigma_ts = sigma_eigs / (u * np.log(u)**2)
        return sigma_ts
