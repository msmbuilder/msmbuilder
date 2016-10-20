from __future__ import print_function, division, absolute_import
from six import PY2
import numpy as np
from msmbuilder.decomposition.tica import tICA
from msmbuilder.decomposition._speigh import scdeflate

__all__ = ['KSparseTICA']


class KSparseTICA(tICA):
    """Sparse tICA where eigenvectors must have k or fewer nonzero values.

    A special case of approximate time-structure Independent Component Analysis (tICA),
    where each approximate eigenvector is required to have k or fewer nonzero values.

    This case has the nice properties that:
    - It may be intuitive for the user to choose `k` rather than continuous parameters for regularization strength.
    - k-sparse approximate eigenvectors can be recovered using the truncated power method, which has no additional
        hyperparameters and typically converge quickly
    - We recover standard tICA when k = n_features

    Here's what's happening under the hood:
    0. Turn the generalized eigenvalue problem A v = u B v into an eigenvalue problem C v = u C v
        A = self.offset_correlation_; B = self.covariance_; C = B^{-1} A;

    1. Recover the dominant* k-sparse eigenvector of C using the
        truncated power method [1]

    2. Remove the "influence" of v from C by Schur's complement deflation
        C = (C - (C x) * (x C)) / (x C x)

    3. Repeat 1,2 until the desired number of components has been recovered.

    .. warning::
        This model is currently  experimental, and may undergo significant
        changes or bug fixes in upcoming releases.

    .. warning::
        The pseudo-eigenpairs can be recovered in the wrong order, i.e. the pseudo-eigenvalue
        of component 5 might be greater than the pseudo-eigenvalue of component 4.
        To recover the top-5 eigenpairs, you should really recover all of the eigenpairs, then sort them.

    .. warning::
        Sometimes the pseudoeigenvalues can exceed 1.
        It's unclear why this happens...

    .. note::
        Haven't tested different deflation methods here yet.

    Parameters
    ----------
    n_components : int
        Number of sparse tICs to find.

    lag_time : int
        Time-lagged correlations are computed between X[t] and X[t+lag_time].

    k : int
        Desired sparsity of components.

    shrinkage : float, default=None
        The covariance shrinkage intensity (range 0-1).

    kinetic_mapping: boolean
        If True, scale the projection onto each eigenvector by its corresponding
        eigenvalues, yielding the kinetic distance as described in [4].

    Attributes
    ----------
    components_ : array-like, shape (n_components, n_features)
        Components with maximum autocorrelation.

    offset_correlation_ : array-like, shape (n_features, n_features)
        Symmetric time-lagged correlation matrix, `C=E[(x_t)^T x_{t+lag}]`.

    eigenvalues_ : array-like, shape (n_features,)
        Psuedo-eigenvalues of the tICA generalized eigenproblem, in decreasing
        order.

    eigenvectors_ : array-like, shape (n_components, n_features)
        Sparse psuedo-eigenvectors of the tICA generalized eigenproblem. The
        vectors give a set of "directions" through configuration space along
        which the system relaxes towards equilibrium.

    means_ : array, shape (n_features,)
        The mean of the data along each feature

    n_observations_ : int
        Total number of data points fit by the model. Note that the model
        is "reset" by calling `fit()` with new sequences, whereas
        `partial_fit()` updates the fit with new data, and is suitable for
        online learning.

    n_sequences_ : int
        Total number of sequences fit by the model. Note that the model
        is "reset" by calling `fit()` with new sequences, whereas
        `partial_fit()` updates the fit with new data, and is suitable for
         online learning.

    timescales_ : array-like, shape (n_components,)
        The implied timescales of the tICA model, given by
        -offset / log(eigenvalues)


    See Also
    --------
    msmbuilder.decomposition.tICA
    msmbuilder.decomposition.SparseTICA

    References
    ----------
    .. [1] Yuan, X-T. and Zhang, T. "Truncated Power Method for Sparse Eigenvalue Problems."
        Journal of Machine Learning Research. Vol. 14. 2013.
        http://www.jmlr.org/papers/volume14/yuan13a/yuan13a.pdf
    .. [2] McGibbon, R. T. and Pande, V.S. "Identification of simple reaction
        coordinates from complex dynamics." arXiv:1602.08776 [cond-mat.stat-mech]
        2016. https://arxiv.org/abs/1602.08776
    .. [3] Mackey, L. "Deflation Methods for Sparse PCA." NIPS. Vol. 21. 2008.
        http://web.stanford.edu/~lmackey/papers/deflation-nips08.pdf
    .. [4] Noe, F. and Clementi, C. "Kinetic distance and kinetic maps from molecular dynamics
        simulation." arXiv:1506.06259 [physics.comp-ph] https://arxiv.org/abs/1506.06259
    """

    def __init__(self, n_components=None, lag_time=1, k=5, shrinkage=None, kinetic_mapping=True):
        super(KSparseTICA, self).__init__(n_components, lag_time=lag_time,
                                          kinetic_mapping=kinetic_mapping, shrinkage=shrinkage)
        self.k = k

    def _truncate(self, x, k):
        ''' given a vector x, leave its top-k absolute-value entries alone, and set the rest to 0 '''
        not_F = np.argsort(np.abs(x))[:-k]
        x[not_F] = 0
        return x

    def _normalize(self, x):
        ''' given a vector x, normalize it '''
        return x / np.linalg.norm(x)

    def _truncated_power_method(self, A, x0, k, max_iter=10000, thresh=1e-8):
        '''
        given a matrix A, an initial guess x0, and a maximum cardinality k,
        find the best k-sparse approximation to its dominant eigenvector

        References
        ----------
        [1] Yuan, X-T. and Zhang, T. "Truncated Power Method for Sparse Eigenvalue Problems."
        Journal of Machine Learning Research. Vol. 14. 2013.
        http://www.jmlr.org/papers/volume14/yuan13a/yuan13a.pdf
        '''

        xts = [x0]
        for t in range(max_iter):
            xts.append(self._normalize(self._truncate(np.dot(A, xts[-1]), k)))
            if np.linalg.norm(xts[-1] - xts[-2]) < thresh: break
        return xts[-1]

    def _solve(self):

        # allow to reduce n_components without re-fitting
        if not self._is_dirty:
            if len(self._eigenvalues_) >= self.n_components:
                return

        # initialize
        A, B = self.offset_correlation_, self.covariance_
        BinvA = np.linalg.inv(B).dot(A)
        eig_vecs = []

        # recover components
        for i in range(self.n_components):
            x0 = self._normalize(np.random.rand(len(A)))
            eig_vecs.append(self._truncated_power_method(BinvA, x0, k=self.k))
            BinvA = scdeflate(BinvA, eig_vecs[-1])
        eig_vecs = np.vstack(eig_vecs)

        # compute pseudoeigenvalues
        vs = eig_vecs.T
        eig_vals = np.diag((vs.T.dot(A).dot(vs)).dot(np.linalg.inv(vs.T.dot(B).dot(vs))))

        # sort
        argsorted = np.argsort(eig_vals)[::-1]
        self._eigenvalues_ = np.zeros((self.n_components))
        self._eigenvectors_ = np.zeros((self.n_features, self.n_components))

        for i in range(self.n_components):
            self._eigenvalues_[i] = eig_vals[argsorted[i]]
            self._eigenvectors_[:, i] = eig_vecs[argsorted[i]]

        self._is_dirty = False

    def summarize(self, n_timescales_to_report=5):
        """Some summary information."""
        nonzeros = np.sum(np.abs(self.eigenvectors_) > 0, axis=0)
        active = '[%s]' % ', '.join(['%d/%d' % (n, self.n_features) for n in nonzeros[:n_timescales_to_report]])

        return """K-sparse time-structure Independent Components Analysis (tICA)
------------------------------------------------------------------
n_components        : {n_components}
shrinkage           : {shrinkage}
lag_time            : {lag_time}
kinetic_mapping     : {kinetic_mapping}
n_features          : {n_features}
Top {n_timescales_to_report} timescales :
{timescales}
Top {n_timescales_to_report} eigenvalues :
{eigenvalues}
Number of active degrees of freedom:
{active}
""".format(n_components=self.n_components, shrinkage=self.shrinkage_, lag_time=self.lag_time,
           kinetic_mapping=self.kinetic_mapping,
           timescales=self.timescales_[:n_timescales_to_report], eigenvalues=self.eigenvalues_[:n_timescales_to_report],
           n_features=self.n_features, active=active, n_timescales_to_report=n_timescales_to_report)