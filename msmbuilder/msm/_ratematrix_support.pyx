# This file is designed to be included by _ratematrix.pyx

cdef eigK(const double[:, ::1] A, npy_intp n, double[::1] pi=None, which='K'):
    """Diagonalize the rate matrix

    If which == 'K', the first argument should be the rate matrix, K, and `pi`
    is ignored. If which == 'S', the first argument should be the symmetric
    rate matrix, S. This can be build using buildK(... which='S'), and pi
    should contain the equilibrium distribution (left eigenvector of K with
    eigenvalue 0, and also the last n elements of exptheta).

    Whichever is supplied the return value is the eigen decomposition of `K`.
    The eigendecomposition of S is not returned.

    Returns
    -------
    w : array
        The eigenvalues of K
    U : array, size=(n,n)
        The left eigenvectors of K
    V : array, size=(n,n)
        The right eigenvectors of K
    """
    cdef npy_intp i, j
    cdef double norm
    cdef double[::1] w
    cdef double[::1, :] U, V, VS
    U = zeros((n, n), order='F')
    V = zeros((n, n), order='F')

    if which == 'S':
        w, VS = scipy.linalg.eigh(A)
        with nogil:
            for j in range(n):
                for i in range(n):
                    V[i, j] = sqrt(pi[j] / pi[i]) * VS[i, j]
                    U[i, j] = sqrt(pi[i] / pi[j]) * VS[i, j]
            for i in range(n):
                cdnrm2(V[:, i], &norm)
                for j in range(n):
                    V[j, i] /= norm
                    U[j, i] *= norm

    else:
        w_, U_, V_ = scipy.linalg.eig(A, left=True, right=True)
        w = ascontiguousarray(real(w_))
        U = asfortranarray(real(U_))
        V = asfortranarray(real(V_))

        with nogil:
            for i in range(n):
                # we need to ensure the proper normalization
                cddot(U[:, i], V[:, i], &norm)
                for j in range(n):
                    U[j, i] = U[j, i] / norm

    if DEBUG:
        assert np.allclose(scipy.linalg.inv(V).T, U)

    return w, U.copy(), V.copy()


cdef int hadamard_X(const double[::1] w, const double[::1] expwt, double t,
                    npy_intp n, double[:, ::1] A) nogil:
    """
    Overwrite the matrix A by the elementwise product of A with the matrix
    X, where :math:`x_{ij}` is:

        if i != j:
            x_{ij} = (e^{t w_i} - e^{t w_j}) / (w_i - w_j)
        else:
            x_{ii} = t * e^{t w_i}

    """
    cdef npy_intp i, j

    for i in range(n):
        for j in range(n):
            if i != j:
                A[i, j] *= (expwt[i] - expwt[j]) / (w[i] - w[j])
            else:
                A[i, j] *= t * expwt[i]


cdef int hadamard_inplace(const double[:, ::1] A, const double[:, ::1] B) nogil:
    """Overwrite the matrix A by its element-wise product with matrix B
    """
    if (A.shape[0] != B.shape[0]) or (A.shape[1] != B.shape[1]):
        return -1

    for i in range(A.shape[0]):
        for j in range(A.shape[1]):
            A[i, j] = A[i, j] * B[i, j]

    return 1;


cdef int transmat(const double[::1] expwt, const double[:, ::1] U,
                  const double[:, ::1] V, npy_intp n,
                  double[:, ::1] temp, double[:, ::1] T) nogil:
    """Compute the transition matrix, expm(Kt), from the eigen-decomposition
    of K

    On exit, T is written into the
    """
    cdef npy_intp i, j
    # T = np.dot(np.dot(V, np.diag(expwt)), U.T)
    for i in range(n):
        for j in range(n):
            temp[i, j] = V[i, j] * expwt[j]
    cdgemm_NT(temp, U, T)


cpdef int dw_du(const double[:, ::1] dKu, const double[:, ::1] V,
            const double[:, ::1] U, npy_intp n, double[::1] temp,
            double[::1] out) nogil:
    r"""Calculate the derivative of the eigenvalues, w, of a matrix, K(\theta),
    with respect to \theta_u.

    Parameters
    ----------
    dKu : array, shape=(n, n)
        Derivative of the rate matrix, K(\theta), with respect to \theta_u
    U : array, shape=(n, n)
        Left eigenvectors of the rate matrix, K(\theta)
    V : array, shape=(n, n)
        Right eigenvectors of the rate matrix, K(\theta)
    n : int
        Size of the matrices
    temp : array, shape=(n,)
        Temporary storage (overwritten)

    Returns
    -------
    out : array, shape=(n,)
        On exit, out[i] contains the derivative of the `i`th eigenvalue
        of K with respect to \theta_u.
    """
    cdef npy_intp i
    for i in range(n):
        cdgemv_N(dKu, U[:, i], temp)
        cddot(temp, V[:, i], &out[i])



cdef dT_dtheta(const double[::1] w, const double[:, ::1] U, const double[:, ::1] V,
               const double[:, ::1] counts, npy_intp n, double t, double[:, ::1] T,
               double[:, ::1] dT):
    """Compute some of the terms required for d(exp(K))/d(theta).

    Returns
    -------
    """
    cdef npy_intp i, j
    cdef double[::1] expwt
    cdef double[:, ::1] X, temp1, temp2, dLdK
    temp1 = zeros((n, n))
    temp2 = zeros((n, n))
    dLdK = zeros((n, n))
    expwt = zeros(n)

    with nogil:
        for i in range(n):
            expwt[i] = exp(w[i]*t)

        transmat(expwt, U, V, n, temp1, T)

        # dLdK[i,j] = counts[i,j] / T[i,j]
        for i in range(n):
            for j in range(n):
                if counts[i, j] > 0:
                    dLdK[i, j] = counts[i, j] / T[i, j]

        # out = U \left(V^T dLdK U \circ X(\lambda, t))\right) V^T

        # temp2 = V^T dLdK U
        cdgemm_TN(V, dLdK, temp1)
        cdgemm_NN(temp1, U, temp2)

        # temp2 =  (V^T dLdK U \circ X(w, t))
        hadamard_X(w, expwt, t, n, temp2)

        # dT = U \left(V^TCU \circ X(\lambda, t))\right) V^T
        cdgemm_NN(U, temp2, temp1)
        cdgemm_NT(temp1, V, dT)

    if DEBUG:
        X = np.subtract.outer(expwt, expwt) / np.subtract.outer(w, w)
        np.fill_diagonal(np.asarray(X), t*np.asarray(expwt))
        Y = np.asarray(U).dot(np.asarray(V).T.dot(dLdK).dot(U) * X).dot(np.asarray(V).T)

        assert np.allclose(dT, Y)
        assert np.allclose(T, np.dot(np.dot(V, np.diag(expwt)), U.T))
        assert np.allclose(T, scipy.linalg.expm(t*np.asarray(K)))
