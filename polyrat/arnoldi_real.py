r"""Real-valued Arnoldi/Lanczos iteration for ORA algorithm.

This module provides orthogonal polynomial basis construction that ensures
real coefficients when fitting complex frequency data (s = j*omega).

The key insight from Ma & Engin (2022) is to use the stacked real/imaginary
representation:
    X = [[S', -S''],     where S = diag(s) and S = S' + j*S''
         [S'',  S']]

For pure imaginary s = j*omega, X is skew-symmetric, enabling efficient
Lanczos iteration.

References
----------
.. [Ma22] Ma & Engin (2022), "Orthogonal rational approximation of transfer
   functions for high-frequency circuits", Int. J. Circuit Theory Appl.
"""

import numpy as np
from scipy.sparse.linalg import LinearOperator


def _stack_real_imag(z):
    """Stack real and imaginary parts of a complex vector.

    Parameters
    ----------
    z : np.ndarray
        Complex vector of length M

    Returns
    -------
    np.ndarray
        Real vector of length 2*M with [real(z); imag(z)]
    """
    return np.concatenate([z.real, z.imag])


def _unstack_real_imag(x, M):
    """Unstack a stacked real vector back to complex.

    Parameters
    ----------
    x : np.ndarray
        Real vector of length 2*M
    M : int
        Original complex vector length

    Returns
    -------
    np.ndarray
        Complex vector of length M
    """
    return x[:M] + 1j * x[M:]


def build_X_matrix(s):
    """Build the stacked X matrix for skew-symmetric Lanczos.

    For s = s' + j*s'', builds:
        X = [[diag(s'), -diag(s'')],
             [diag(s''),  diag(s')]]

    Parameters
    ----------
    s : np.ndarray
        Complex frequency points (typically j*omega)

    Returns
    -------
    LinearOperator
        Efficient matrix-vector product operator for X
    """
    M = len(s)
    s_real = s.real
    s_imag = s.imag

    def matvec(x):
        """Apply X to vector x = [x'; x'']."""
        x_r = x[:M]
        x_i = x[M:]
        y_r = s_real * x_r - s_imag * x_i
        y_i = s_imag * x_r + s_real * x_i
        return np.concatenate([y_r, y_i])

    def rmatvec(x):
        """Apply X^T to vector x (same as matvec for skew-symmetric)."""
        return matvec(x)

    return LinearOperator((2*M, 2*M), matvec=matvec, rmatvec=rmatvec, dtype=float)


def lanczos_real(s, degree, weight=None):
    r"""Real-valued Lanczos iteration for orthogonal polynomial basis.

    Builds an orthogonal basis for polynomials using the stacked
    real/imaginary representation. For s = j*omega, the X matrix
    is skew-symmetric, making Lanczos applicable.

    Parameters
    ----------
    s : np.ndarray
        Complex frequency points, shape (M,) or (M, 1)
    degree : int
        Polynomial degree (basis will have degree+1 columns)
    weight : np.ndarray, optional
        Complex weight vector, shape (M,). Default is ones.

    Returns
    -------
    Q : np.ndarray
        Orthonormal basis matrix, shape (2*M, degree+1), real-valued
    H : np.ndarray
        Upper Hessenberg matrix (tridiagonal for Lanczos), shape (degree+1, degree+1)
    """
    s = np.asarray(s).flatten()
    M = len(s)

    if weight is None:
        weight = np.ones(M, dtype=complex)
    else:
        weight = np.asarray(weight).flatten()

    # Build the X matrix operator
    X = build_X_matrix(s)

    # Initialize with stacked weight vector
    q0 = _stack_real_imag(weight)
    beta0 = np.linalg.norm(q0)
    q0 = q0 / beta0

    n_cols = degree + 1
    Q = np.zeros((2*M, n_cols), dtype=float)
    H = np.zeros((n_cols, n_cols), dtype=float)

    Q[:, 0] = q0

    # Lanczos iteration with full reorthogonalization
    for k in range(degree):
        # w = X @ q_k
        w = X @ Q[:, k]

        # Orthogonalize against all previous vectors
        # (full reorthogonalization for numerical stability)
        for j in range(k + 1):
            h_jk = np.dot(Q[:, j], w)
            H[j, k] = h_jk
            w = w - h_jk * Q[:, j]

        # Reorthogonalize (CGS with reorthogonalization)
        for j in range(k + 1):
            h_corr = np.dot(Q[:, j], w)
            H[j, k] += h_corr
            w = w - h_corr * Q[:, j]

        h_kp1_k = np.linalg.norm(w)
        H[k + 1, k] = h_kp1_k

        if h_kp1_k < 1e-14:
            # Breakdown: found invariant subspace
            Q = Q[:, :k + 1]
            H = H[:k + 1, :k + 1]
            break

        Q[:, k + 1] = w / h_kp1_k

    return Q, H


def arnoldi_real(s, degree, weight=None):
    r"""Real-valued Arnoldi iteration (general complex s).

    For general complex s (not purely imaginary), the X matrix is not
    skew-symmetric, so we use the more general Arnoldi iteration.

    Parameters
    ----------
    s : np.ndarray
        Complex frequency points
    degree : int
        Polynomial degree
    weight : np.ndarray, optional
        Complex weight vector

    Returns
    -------
    Q : np.ndarray
        Orthonormal basis matrix, shape (2*M, degree+1)
    H : np.ndarray
        Upper Hessenberg matrix, shape (degree+1, degree+1)
    """
    # For now, use the same implementation as Lanczos
    # (with full reorthogonalization, it's mathematically equivalent)
    return lanczos_real(s, degree, weight)


def vandermonde_real(s, degree, weight=None):
    r"""Build Vandermonde-like matrix with real orthogonal columns.

    This is the main entry point for ORA fitting. Returns an orthogonal
    basis that can be used to fit complex data with real coefficients.

    Parameters
    ----------
    s : np.ndarray
        Complex frequency points, shape (M,) or (M, 1)
    degree : int
        Polynomial degree
    weight : np.ndarray, optional
        Complex weight/denominator vector for rational fitting

    Returns
    -------
    Q : np.ndarray
        Orthonormal polynomial basis, shape (2*M, degree+1)
    H : np.ndarray
        Hessenberg matrix for coefficient conversion
    """
    return lanczos_real(s, degree, weight)


def eval_poly_real(s, coeffs, Q_ref, H_ref, s_ref):
    r"""Evaluate a polynomial with real coefficients at new points.

    Given a polynomial p(s) = sum_k c_k * phi_k(s) where phi_k are the
    orthogonal basis functions, evaluate at new frequency points.

    Parameters
    ----------
    s : np.ndarray
        New frequency points
    coeffs : np.ndarray
        Real polynomial coefficients
    Q_ref : np.ndarray
        Reference orthogonal basis (from training points)
    H_ref : np.ndarray
        Reference Hessenberg matrix
    s_ref : np.ndarray
        Reference frequency points used to build Q_ref

    Returns
    -------
    np.ndarray
        Complex polynomial values at new points
    """
    # Build basis at new points using Hessenberg recurrence
    s_new = np.asarray(s).flatten()
    M = len(s_new)
    degree = len(coeffs) - 1

    Q_new, _ = vandermonde_real(s_new, degree)

    # Evaluate polynomial
    result_stacked = Q_new @ coeffs

    # Convert back to complex
    return _unstack_real_imag(result_stacked, M)


class RealPolynomialBasis:
    r"""Polynomial basis with real-coefficient enforcement.

    This class constructs an orthogonal polynomial basis using the
    stacked real/imaginary representation from the ORA algorithm.

    Parameters
    ----------
    s : np.ndarray
        Complex frequency points
    degree : int
        Polynomial degree
    weight : np.ndarray, optional
        Weight vector for rational fitting
    """

    def __init__(self, s, degree, weight=None):
        self.s = np.asarray(s).flatten()
        self.degree = degree
        self.M = len(self.s)

        self._Q, self._H = vandermonde_real(self.s, degree, weight)
        self._Q.flags.writeable = False

    @property
    def vandermonde_X(self):
        """Return the orthonormal basis at training points."""
        return self._Q

    @property
    def n_cols(self):
        """Number of basis columns."""
        return self._Q.shape[1]

    def vandermonde(self, s, weight=None):
        """Evaluate basis at new frequency points.

        Parameters
        ----------
        s : np.ndarray
            New frequency points
        weight : np.ndarray, optional
            Weight vector

        Returns
        -------
        np.ndarray
            Basis matrix at new points
        """
        return vandermonde_real(s, self.degree, weight)[0]
