
import numpy as np
import scipy.linalg
from mdtraj import io
from . import tICA
from sklearn.metrics.pairwise import pairwise_kernels
from ..base import BaseEstimator
from sklearn.base import TransformerMixin

class ktICA(BaseEstimator, TransformerMixin):
    r""" 
    Time-structure Independent Componenent Analysis (tICA) using the kernel
    trick. 

    The kernel trick allows one to extend a linear method (e.g. tICA) to
    include non-linear solutions. 

    <Add some exposition here>

    Parameters
    ----------
    kernel : str or callable
        The kernel function to define similarities in the feature space.
        It must be one of:
            - 'linear' : linear kernel (dot product in the input space)
            - 'poly' : polynomial kernel
            - 'rbf' : radial basis function 
            - 'sigmoid' :
            - 'precomputed' : the precomputed gram matrix will be passed
                              to ktICA.fit().
            - callable : function that takes two datasets and returns
                         a matrix of similarities
    degree : int, optional
        Degree of the polynomial kernel. This is only used if kernel
        is 'poly'.
    gamma : float, optional
        Kernel coefficient for 'rbf', 'poly', and 'sigmoid'. If gamma == 0.0,
        then will use 1 / n_features.
    coef0 : float, optional
        Independent term in 'poly' and 'sigmoid'.
    lag_time : int, optional
        Lag time to define the offset correlation matrix.
    stride : int, optional
        Only sample pairs of points from the data according to this stride
    eta : float, optional
        Regularization strength for solving the ktICA problem
    n_components : int, optional
        Number of components to project onto in the `transform` method
        and to use when scoring.

    Attributes
    ----------
    components_ : array-like, shape (n_components, n_datapoints)
        Components with maximum autocorrelation; n_datapoints is equal
        to twice the number of pairs of points used to fit the model.
    eigenvalues_ : array-like, shape (n_datapoints,)
        Eigenvalues of the tICA generalized eigenproblem, in decreasing
        order.
    eigenvectors_ : array-like, shape (n_datapoints, n_datapoints)
        ALLT Eigenvectors of the tICA generalized eigenproblem. The vectors
        give a set of "directions" through configuration space along
        which the system relaxes towards equilibrium. Each eigenvector
        is associated with characteristic timescale
        :math:`- \frac{lag_time}{ln \lambda_i}, where :math:`lambda_i` is
        the corresponding eigenvector. 
        The eigenvectors are stored in columns and normalized to have
        unit variance according to the training data.
    uncentered_gram_matrix_ : array-like, shape (n_datapoints, n_datapoints)
        The uncentered gram matrix of similarities.
    n_datapoints_ : int
        Total number of data points fit by the model. This is equal to
        two times the number of pairs sampled during fitting.
    timescales_ : array-like, shape (n_components,)
        The implied timescales of the tICA model, given by
        - lag_time / log(eigenvalues)

    Notes
    -----
    There are many ways to approximate the kernel solution to the tICA
    problem that are not nearly as computationally intense as this 
    method. Checkout sklearn.kernel_approximation for details.

    References
    ----------
    .. [1] Schwantes, Christian R. and Vijay S. Pande. "Modeling Molecular
           Kinetics with tICA and the Kernel Trick." In Review. (2014)
    """
    _available_kernels = ['rbf', 'sigmoid', 'linear', 'poly'] 
    
    def __init__(self, kernel='rbf', degree=3, gamma=1.0, coef0=0.0, 
                 lag_time=1, stride=1, n_components=1, eta=1.0):

        if not kernel in self._available_kernels:
            if kernel != 'precomputed':
                if not callable(kernel):
                    raise ValueError("kernel must be one of %s or 'precomputed' or a callable function" % str(self._available_kernels))

        self.kernel = kernel

        self.degree = int(degree)
        self.gamma = float(gamma)
        self.coef0 = float(coef0)
        self.eta = float(eta)

        self.lag_time = int(lag_time)
        self.stride = int(stride)
        self.n_components = int(n_components)

    @property
    def components_(self):
        return self.eigenvectors_[:, :self.n_components].T


    @property
    def timescales_(self):
        return - self.lag_time / np.log(self.eigenvalues_[:self.n_components])


    @property
    def _kernel_params(self):
        return {'degree' : self.degree, 'gamma' : self.gamma, 
                'coef' : self.coef0}


    def fit(self, sequences):
        r"""
        Fit the model to a given timeseries
        
        Parameters
        ----------
        sequences : array-like, shape = [n_sequences, n_samples, n_features]
            A set of sequences to fit the model to.           
            If the kernel you specified is 'precomputed', then X should
            be the UNCENTERED gram matrix of inner products.

        Returns
        -------
        self : object
            Returns the instance itself.
        """

        if self.kernel == 'precomputed':
            if X.shape[0] != X.shape[1]:
                raise ValueError("X is supposed to be square for kernel='precomputed'")

            self.uncentered_gram_matrix_ = X
            self.__Xfit = None
            # this will throw errors if we try to access this later
            self.n_datapoints_ = self.uncentered_gram_matrix_.shape[0]

        else:
            # sequences holding the subsampled points
            X_0 = []
            X_t = []

            for seq in sequences:
                seq_t = seq[self.lag_time::self.stride]
                seq_0 = seq[::self.stride][:len(seq_t)]
                X_0.append(seq_0)
                X_t.append(seq_t)

            self.__Xfit = np.concatenate([np.concatenate(X_0), np.concatenate(X_t)])
            self.n_datapoints_ = len(self.__Xfit)

            self.uncentered_gram_matrix_ = pairwise_kernels(self.__Xfit, metric=self.kernel,
                                                            n_jobs=1, filter_params=True,
                                                            **self._kernel_params)           

        # just make sure it's actually symmetric
        self.uncentered_gram_matrix = (self.uncentered_gram_matrix_ + self.uncentered_gram_matrix_.T) * 0.5

        # matrix used when centering
        oneN = np.ones(self.n_datapoints_) / float(self.n_datapoints_)
        oneN.reshape((-1, 1))

        self.gram_matrix_ = self.uncentered_gram_matrix_ \
                            - oneN.T.dot(self.uncentered_gram_matrix_) \
                            - self.uncentered_gram_matrix_.dot(oneN) \
                            + oneN.T.dot(self.uncentered_gram_matrix_.dot(oneN))

        # again just make sure that there's no rounding issues
        self.gram_matrix_ = (self.gram_matrix_ + self.gram_matrix_.T) * 0.5

        n_pairs = self.n_datapoints_ / 2
        R = np.zeros((self.n_datapoints_, self.n_datapoints_))
        R[:n_pairs, n_pairs:] = np.eye(n_pairs)
        R[n_pairs:, :n_pairs] = np.eye(n_pairs)

        KK = self.gram_matrix_.dot(self.gram_matrix_)

        lhs = self.gram_matrix_.dot(R).dot(self.gram_matrix_)
        rhs = KK + self.eta * np.eye(self.n_datapoints_)

        vals, vecs = scipy.linalg.eigh(lhs, b=rhs)

        dec_ind = np.argsort(vals)[::-1]
        
        self.eigenvalues_ = vals[dec_ind]
        vecs = vecs[:, dec_ind]

        # now, normalize the eigenvectors to have unit variance
        vKK = vecs.T.dot(KK)
        # not sure if I should compute the variance based on
        # the regularization strength or not :/
        vec_vars = np.sum(vKK * vecs.T, axis=1) / self.n_datapoints_
        self.eigenvectors_ = vecs / np.sqrt(vec_vars)

        return self


    def transform(self, sequences):
        """
        Project a point onto the top `n_components` ktICs

        Parameters
        ----------
        sequences : np.ndarray, shape = [n_sequences, n_points, n_features]
            Data to project onto eigenvector. If kernel == 'precomputed'
            Then this must be a kernel matrix, where:
                X[i, j] = kernel(Xfit[i], Xnew[j]) 
            where Xfit are all of the points used to fit the model.

        Returns
        -------
        Xnew : np.ndarray, shape = [n_sequences, n_points, n_components]
            Projected value of each point in each trajectory
        """
        sequence_new = []
        for Y in sequences:
            Ku = pairwise_kernels(self.__Xfit, Y=Y, metric=self.kernel,
                                  filter_params=True, n_jobs=1, 
                                  **self._kernel_params)
            # rows are training points, columns are test points

            oneN = np.ones((self.n_datapoints_, 1)) / float(self.n_datapoints_)

            # center this using the mean from the training set
            K = Ku - self.uncentered_gram_matrix_.dot(oneN) \
                - oneN.T.dot(Ku) \
                + oneN.T.dot(self.uncentered_gram_matrix_.dot(oneN))

            Xnew = (self.components_.dot(K)).T
            sequence_new.append(Xnew)
        return sequence_new


    def score(self, sequences, y=None):
        """
        Compute the GMRQ for this model given some unobserved data.

        Parameters
        ----------
        sequences : list of array-like
            List of sequences to use for scoring this model. If 
            kernel is 'precomputed', then this must be a gram 
            matrix of similarities between the new data (in columns)
            and the data used to fit the model (in rows)

        Returns
        -------
        gmrq : float
            Generalized matrix Rayleigh quotient. This number indicates how
            well the top ``n_timescales`` ktICs of this model perform as
            slowly decorrelating collective variables for the new data in
            ``sequences``.

        References
        ----------
        .. [1] McGibbon, R. T. and V. S. Pande, "Variational cross-validation
           of slow dynamical modes in molecular kinetics"
           http://arxiv.org/abs/1407.8083 (2014)
        """
        sequence_new = self.transform(sequences)

        # We can use the tICA machinery to compute the autocorrelations
        # we need.
        tica_obj = tICA(lag_time=self.lag_time, gamma=0.0)
        tica_obj.fit(sequence_new)
        trace = tica_obj.eigenvalues_.sum()
        trace2 = np.trace(tica_obj.offset_correlation_.dot(np.linalg.inv(tica_obj.covariance_)))
        
        assert trace == trace2

        return trace
