r"""Orthogonal Rational Approximation (ORA) with real coefficient enforcement.

This module implements the ORA algorithm from Ma & Engin (2022) for fitting
rational functions with real polynomial coefficients to complex frequency data.

The implementation follows the same structure as StabilizedSKRationalApproximation
for consistency with the rest of polyrat.

References
----------
.. [Ma22] Ma & Engin (2022), "Orthogonal rational approximation of transfer
   functions for high-frequency circuits", Int. J. Circuit Theory Appl.
"""

import numpy as np
import scipy.linalg
from iterprinter import IterationPrinter

from .arnoldi_real import (
    RealPolynomialBasis, vandermonde_real,
    _stack_real_imag, _unstack_real_imag
)
from .rational import RationalApproximation, RationalRatio
from .polynomial import Polynomial


def _flip_unstable_poles(poles):
    """Flip poles with positive real part to ensure stability."""
    poles = np.asarray(poles, dtype=complex).copy()
    unstable = poles.real > 0
    poles[unstable] = -np.abs(poles[unstable].real) + 1j * poles[unstable].imag
    return poles


def _stacked_complex_multiply(f_stacked, q_stacked, M):
    """Multiply stacked complex vectors element-wise.

    For f = f' + jf'' and q = q' + jq'', computes f*q in stacked form.
    """
    f_real = f_stacked[:M]
    f_imag = f_stacked[M:]
    q_real = q_stacked[:M]
    q_imag = q_stacked[M:]

    result_real = f_real * q_real - f_imag * q_imag
    result_imag = f_real * q_imag + f_imag * q_real

    return np.concatenate([result_real, result_imag])


def minimize_2norm_varpro_real(P, Q, y_stacked, M):
    """Variable projection for stacked real representation.

    Solves: min ||P @ a - F @ Q @ b||_2
    where F represents element-wise complex multiplication with y.

    Parameters
    ----------
    P : np.ndarray (2M, num_deg+1)
        Numerator basis (stacked real)
    Q : np.ndarray (2M, denom_deg+1)
        Denominator basis (stacked real)
    y_stacked : np.ndarray (2M,) or (2M, nout)
        Target values (stacked real), single or multi-output
    M : int
        Number of complex frequency points

    Returns
    -------
    a : np.ndarray (num_deg+1,) or (num_deg+1, nout)
        Numerator coefficients
    b : np.ndarray (denom_deg+1,)
        Denominator coefficients (shared across outputs)
    cond : float
        Condition number indicator
    """
    # QR factorization of P for projection
    Q_P, R_P = np.linalg.qr(P, mode='reduced')

    n_denom = Q.shape[1]
    n_num = P.shape[1]

    # Handle multi-output: y_stacked can be (2M,) or (2M, nout)
    y_stacked = np.asarray(y_stacked)
    if y_stacked.ndim == 1:
        nout = 1
        y_list = [y_stacked]
    else:
        nout = y_stacked.shape[1]
        y_list = [y_stacked[:, j] for j in range(nout)]

    # Build projected matrices for all outputs and stack BEFORE SVD
    # This ensures the denominator is fit using all outputs jointly
    A_blocks = []
    for y_j in y_list:
        # Build F @ Q: multiply each column of Q by y_j (in complex domain)
        FQ = np.zeros_like(Q)
        for k in range(n_denom):
            FQ[:, k] = _stacked_complex_multiply(y_j, Q[:, k], M)

        # Project out numerator subspace: A_j = (I - P @ P^T) @ FQ
        A_j = FQ - Q_P @ (Q_P.T @ FQ)
        A_blocks.append(A_j)

    # Stack all outputs for joint denominator fitting
    A = np.vstack(A_blocks)

    # Find denominator coefficients via SVD (null space)
    U, sigma, VH = np.linalg.svd(A, full_matrices=False)

    # Condition number indicator
    with np.errstate(divide='ignore'):
        if len(sigma) >= 2:
            cond = sigma[0] * np.sqrt(2) / (sigma[-2] - sigma[-1] + 1e-16)
        else:
            cond = sigma[0] if len(sigma) > 0 else 1.0

    b = VH[-1, :].real  # Last row of VH (smallest singular value)
    b = b / b[0]  # Normalize

    # Compute numerator coefficients for each output using shared b
    Qb = Q @ b
    if nout == 1:
        FQb = _stacked_complex_multiply(y_list[0], Qb, M)
        a = scipy.linalg.solve_triangular(R_P, Q_P.T @ FQb)
    else:
        a = np.zeros((n_num, nout))
        for j, y_j in enumerate(y_list):
            FQb = _stacked_complex_multiply(y_j, Qb, M)
            a[:, j] = scipy.linalg.solve_triangular(R_P, Q_P.T @ FQb)

    return a, b, cond


def ora_fit_real(s, y, num_degree, denom_degree, maxiter=20, xtol=1e-7,
                 verbose=True, weight=None):
    """ORA fitting with SK iteration structure.

    Parameters
    ----------
    s : np.ndarray (M,) or (M,1)
        Complex frequency points
    y : np.ndarray (M,) or (M, *nout)
        Complex target values
    num_degree, denom_degree : int
        Polynomial degrees
    maxiter : int
        Maximum iterations
    xtol : float
        Convergence tolerance
    verbose : bool
        Print progress
    weight : np.ndarray, optional
        Weighting for residual norm

    Returns
    -------
    numerator : Polynomial
        Numerator polynomial
    denominator : Polynomial
        Denominator polynomial
    hist : list
        Iteration history
    """
    s = np.asarray(s).flatten()
    y = np.asarray(y)
    M = len(s)

    # Handle array-valued y
    y_flat = y.reshape(M, -1)
    nout = y_flat.shape[1]

    if verbose:
        printer = IterationPrinter(
            it='4d', res_norm='21.15e', delta_fit='8.2e', cond='8.2e'
        )
        printer.print_header(
            it='iter', res_norm='residual norm',
            delta_fit='delta fit', cond='cond'
        )

    # Initialize denominator values (complex)
    denom = np.ones(M, dtype=complex)

    # Track best solution
    best_res_norm = np.inf
    best_sol = None
    hist = []

    fit_old = np.zeros_like(y_flat)

    for it in range(maxiter):
        # Build bases with frequency normalization
        # Weight = 1/|denom| for stabilization
        num_basis = RealPolynomialBasis(s, num_degree, weight=1./denom)
        denom_basis = RealPolynomialBasis(s, denom_degree, weight=1./denom)

        P = num_basis.vandermonde_X
        Q = denom_basis.vandermonde_X

        # Stack target values (single or multi-output)
        # For multi-output, all outputs are stacked and solved jointly
        # to ensure the shared denominator is fit using all outputs
        if nout == 1:
            y_stacked = _stack_real_imag(y_flat[:, 0])
        else:
            y_stacked = np.column_stack([_stack_real_imag(y_flat[:, j]) for j in range(nout)])

        a, b, cond = minimize_2norm_varpro_real(P, Q, y_stacked, M)

        # Evaluate fit
        Pa = P @ a
        Qb = Q @ b

        # Convert to complex
        if nout == 1:
            Pa_complex = _unstack_real_imag(Pa, M)
            Qb_complex = _unstack_real_imag(Qb, M)
            fit = Pa_complex / Qb_complex
            fit = fit.reshape(-1, 1)
        else:
            Qb_complex = _unstack_real_imag(Qb, M)
            fit = np.zeros((M, nout), dtype=complex)
            for j in range(nout):
                Pa_j = _unstack_real_imag(P @ a[:, j], M)
                fit[:, j] = Pa_j / Qb_complex

        # Compute residual
        if weight is not None:
            w = np.asarray(weight).reshape(-1, 1)
            res_norm = np.linalg.norm(w * (fit - y_flat), 'fro')
            delta_fit = np.linalg.norm(w * (fit - fit_old), 'fro')
        else:
            res_norm = np.linalg.norm(fit - y_flat, 'fro')
            delta_fit = np.linalg.norm(fit - fit_old, 'fro')

        # Track best
        if res_norm < best_res_norm:
            best_res_norm = res_norm
            best_sol = (
                Polynomial(num_basis, a.flatten() if nout == 1 else a),
                Polynomial(denom_basis, b)
            )

        hist.append({'fit': fit.copy(), 'cond': cond, 'res_norm': res_norm})

        if verbose:
            printer.print_iter(it=it, res_norm=res_norm, delta_fit=delta_fit, cond=cond)

        if delta_fit < xtol:
            break

        # Update denominator weights
        denom = np.abs(denom * Qb_complex)
        denom[denom == 0] = 1.0
        fit_old = fit.copy()

    if best_sol is None:
        # Didn't find any valid solution
        best_sol = (
            Polynomial(num_basis, a.flatten() if nout == 1 else a),
            Polynomial(denom_basis, b)
        )

    return best_sol[0], best_sol[1], hist


class RealPolynomial:
    """Wrapper for polynomial with real coefficients in stacked basis."""

    def __init__(self, basis, coef):
        self.basis = basis
        self.coef = np.asarray(coef)

    def __call__(self, s):
        s = np.asarray(s).flatten()
        M = len(s)
        Q = self.basis.vandermonde(s)
        result_stacked = Q @ self.coef
        return _unstack_real_imag(result_stacked, M)

    def roots(self):
        """Compute roots of the polynomial."""
        # Convert to monomial basis and find roots
        s_train = self.basis.s
        M = len(s_train)
        vals = self(s_train)
        degree = len(self.coef) - 1

        if degree == 0:
            return np.array([])

        # Fit monomial polynomial
        omega = s_train.imag  # Assuming s = j*omega
        if np.allclose(s_train.real, 0):
            # Pure imaginary case
            V = np.vander(omega / self.basis.s_scale, degree + 1, increasing=True)
            j_powers = np.array([1j**k for k in range(degree + 1)])
            A = np.vstack([
                (V * j_powers.real).T.real.T,
                (V * j_powers.imag).T.real.T
            ])
            b = np.concatenate([vals.real, vals.imag])
            mono_coeffs, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
            # Scale coefficients back
            for k in range(degree + 1):
                mono_coeffs[k] /= self.basis.s_scale**k
        else:
            V = np.vander(s_train, degree + 1, increasing=True)
            A = np.vstack([V.real, V.imag])
            b = np.concatenate([vals.real, vals.imag])
            mono_coeffs, _, _, _ = np.linalg.lstsq(A, b, rcond=None)

        return np.roots(mono_coeffs[::-1])


class ORARationalApproximation(RationalApproximation):
    r"""Rational approximation with real coefficient enforcement (ORA).

    This class implements the Orthogonal Rational Approximation (ORA)
    algorithm, following the same interface as StabilizedSKRationalApproximation.

    Parameters
    ----------
    num_degree : int
        Numerator polynomial degree
    denom_degree : int
        Denominator polynomial degree
    maxiter : int
        Maximum SK iterations
    xtol : float
        Convergence tolerance
    verbose : bool
        Print progress

    Examples
    --------
    >>> import numpy as np
    >>> from polyrat.ora import ORARationalApproximation
    >>> omega = np.linspace(1e6, 1e9, 100)
    >>> s = 1j * omega
    >>> H = 1 / (1 + s * 1e-9)
    >>> ora = ORARationalApproximation(num_degree=0, denom_degree=1)
    >>> ora.fit(s.reshape(-1,1), H)
    >>> H_fit = ora(s.reshape(-1,1))
    """

    def __init__(self, num_degree, denom_degree, maxiter=20, xtol=1e-7, verbose=True):
        RationalApproximation.__init__(self, num_degree, denom_degree)
        self.maxiter = int(maxiter)
        self.xtol = float(xtol)
        self.verbose = verbose

        self.numerator = None
        self.denominator = None
        self.hist = None

    def fit(self, X, y, weight=None):
        """Fit the rational function to data.

        Parameters
        ----------
        X : np.ndarray (M, 1)
            Complex frequency points
        y : np.ndarray (M,) or (M, *nout)
            Complex function values
        weight : np.ndarray, optional
            Weighting for residual
        """
        X = np.asarray(X)
        y = np.asarray(y)

        # Flatten X if needed
        s = X.flatten()

        self.numerator, self.denominator, self.hist = ora_fit_real(
            s, y, self.num_degree, self.denom_degree,
            maxiter=self.maxiter, xtol=self.xtol, verbose=self.verbose,
            weight=weight
        )

    def eval(self, X):
        """Evaluate the fitted rational function."""
        X = np.asarray(X)
        s = X.flatten()
        M = len(s)

        # Evaluate numerator and denominator using stacked basis
        num_basis = self.numerator.basis
        denom_basis = self.denominator.basis

        # Get Vandermonde at new points (uses same scale as training)
        Qn = num_basis.vandermonde(s)
        Qd = denom_basis.vandermonde(s)

        # Compute denominator values
        denom_stacked = Qd @ self.denominator.coef
        denom_vals = _unstack_real_imag(denom_stacked, M)

        # Compute numerator values (may be multi-output)
        num_coef = self.numerator.coef
        if num_coef.ndim == 1:
            # Single output
            num_stacked = Qn @ num_coef
            num_vals = _unstack_real_imag(num_stacked, M)
            return num_vals / denom_vals
        else:
            # Multi-output
            nout = num_coef.shape[1]
            result = np.zeros((M, nout), dtype=complex)
            for j in range(nout):
                num_stacked = Qn @ num_coef[:, j]
                num_vals = _unstack_real_imag(num_stacked, M)
                result[:, j] = num_vals / denom_vals
            return result

    def __call__(self, X):
        """Evaluate the fitted rational function."""
        return self.eval(X)

    @property
    def poles(self):
        """Return poles of the rational function."""
        if self.denominator is None:
            return None
        poles = self.denominator.roots()
        return _flip_unstable_poles(poles)

    def get_real_coefficients(self):
        """Return numerator and denominator coefficients (both real).

        Returns
        -------
        num_coeffs : np.ndarray
            Real numerator coefficients (in orthogonal basis)
        denom_coeffs : np.ndarray
            Real denominator coefficients (in orthogonal basis)
        """
        return self.numerator.coef, self.denominator.coef
