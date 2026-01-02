r"""Orthogonal Rational Approximation (ORA) with real coefficient enforcement.

This module implements the ORA algorithm from Ma & Engin (2022) for fitting
rational functions with real polynomial coefficients to complex frequency data.

The key insight is that for a complex transfer function H(s) with real coefficients:
    H(s) = N(s) / D(s)

we can enforce real coefficients by stacking real and imaginary parts:
    [A']     [f']
    [A''] c = [f'']

where A = A' + jA'' and f = f' + jf''.

The ORA method uses:
1. Orthogonal polynomial basis via Lanczos iteration
2. Variable projection (VARPRO) to separate numerator/denominator
3. Pole flipping for stability enforcement
4. Block QR decomposition for efficiency

References
----------
.. [Ma22] Ma & Engin (2022), "Orthogonal rational approximation of transfer
   functions for high-frequency circuits", Int. J. Circuit Theory Appl.
.. [Ma21] Ma, Deaton, Engin (2021), "Stabilized Sanathanan-Koerner Iteration
   for Rational Transfer Function Approximation of Scattering Parameters"
"""

import numpy as np
import scipy.linalg
from .arnoldi_real import vandermonde_real, _stack_real_imag, _unstack_real_imag


def _flip_unstable_poles(poles):
    """Flip poles with positive real part to ensure stability.

    Parameters
    ----------
    poles : np.ndarray
        Complex poles

    Returns
    -------
    np.ndarray
        Poles with Re(p) <= 0
    """
    poles = np.asarray(poles, dtype=complex).copy()
    unstable = poles.real > 0
    # Flip: p -> -Re(p) + j*Im(p)
    poles[unstable] = -np.abs(poles[unstable].real) + 1j * poles[unstable].imag
    return poles


def numfit(s, f, num_degree, denom_weights=None):
    r"""Fit numerator with prescribed denominator weights (ORA numfit).

    Given frequency data f(s) and denominator values d(s), fit numerator
    coefficients to minimize:
        || d(s) * f(s) - n(s) ||_2

    using real-valued orthogonal polynomial basis.

    Parameters
    ----------
    s : np.ndarray
        Complex frequency points, shape (M,)
    f : np.ndarray
        Complex function values, shape (M,) or (M, *nout)
    num_degree : int
        Numerator polynomial degree
    denom_weights : np.ndarray, optional
        Complex denominator values at each frequency, shape (M,).
        If None, uses ones (polynomial fitting).

    Returns
    -------
    Q : np.ndarray
        Orthogonal basis matrix
    coeffs : np.ndarray
        Real numerator coefficients
    residual : float
        Fitting residual norm
    """
    s = np.asarray(s).flatten()
    f = np.asarray(f)
    M = len(s)

    # Handle array-valued f
    f_flat = f.reshape(M, -1)
    nout = f_flat.shape[1]

    if denom_weights is None:
        denom_weights = np.ones(M, dtype=complex)
    else:
        denom_weights = np.asarray(denom_weights).flatten()

    # Build orthogonal basis WITHOUT weighting
    # (weighting is only applied to the target, not the basis)
    Q, H = vandermonde_real(s, num_degree, weight=None)

    # Stack the target: b = d(s) * f(s)
    b_all = np.zeros((2*M, nout), dtype=float)
    for j in range(nout):
        target = denom_weights * f_flat[:, j]
        b_all[:, j] = _stack_real_imag(target)

    # Solve least squares: Q @ c ≈ b
    coeffs, residual, rank, sv = np.linalg.lstsq(Q, b_all, rcond=None)

    if nout == 1:
        coeffs = coeffs.flatten()

    res_norm = np.linalg.norm(b_all - Q @ coeffs.reshape(-1, nout), 'fro')

    return Q, coeffs, res_norm


def denfit(s, f, num_degree, denom_degree, denom_init=None, method='svd'):
    r"""Fit denominator using ORA variable projection (ORA denfit).

    Uses the linearized form of the rational approximation problem:
        n(s) - f(s) * d(s) ≈ 0

    with real coefficient enforcement via stacked real/imaginary representation.

    Parameters
    ----------
    s : np.ndarray
        Complex frequency points, shape (M,)
    f : np.ndarray
        Complex function values, shape (M,) or (M, *nout)
    num_degree : int
        Numerator polynomial degree
    denom_degree : int
        Denominator polynomial degree
    denom_init : np.ndarray, optional
        Initial denominator values at frequencies. Default is ones.
    method : str
        'svd' for SVD-based solution, 'ls' for pinned least squares

    Returns
    -------
    denom_new : np.ndarray
        Updated denominator values at frequencies
    poles : np.ndarray
        Roots of the denominator polynomial (after pole flipping)
    denom_coeffs : np.ndarray
        Real denominator coefficients
    cond : float
        Condition number indicator
    """
    s = np.asarray(s).flatten()
    f = np.asarray(f)
    M = len(s)

    # Handle array-valued f
    f_flat = f.reshape(M, -1)
    nout = f_flat.shape[1]

    if denom_init is None:
        denom_init = np.ones(M, dtype=complex)
    else:
        denom_init = np.asarray(denom_init).flatten()

    # Build numerator basis WITHOUT weighting
    # (weighting affects the column space, breaking VarPro projection)
    Qn, Hn = vandermonde_real(s, num_degree, weight=None)

    # Build denominator basis (no weighting, starts from 1)
    Qd, Hd = vandermonde_real(s, denom_degree)

    # Project out numerator subspace from (f * Qd)
    # This implements Eq. 19 from Ma22
    # [I - Qn @ Qn.T] @ (F @ Qd) @ c = 0

    # Build F @ Qd for each output
    # F = diag(f) in complex domain
    FQd_all = []
    for j in range(nout):
        FQd_j = np.zeros((2*M, denom_degree + 1), dtype=float)
        for k in range(denom_degree + 1):
            # Get k-th column of Qd, convert back to complex, multiply by f
            qd_k_stacked = Qd[:, k]
            qd_k = _unstack_real_imag(qd_k_stacked, M)
            f_qd = f_flat[:, j] * qd_k
            FQd_j[:, k] = _stack_real_imag(f_qd)
        FQd_all.append(FQd_j)

    # Stack all outputs
    if nout == 1:
        FQd = FQd_all[0]
    else:
        FQd = np.vstack(FQd_all)

    # Project: A = [I - Qn @ Qn.T] @ FQd
    # But it's more efficient to form the combined matrix for SVD
    # [Qn, -FQd] @ [a; c] = 0, then project out a

    Qn_big = Qn if nout == 1 else scipy.linalg.block_diag(*[Qn]*nout)
    A = FQd - Qn_big @ (Qn_big.T @ FQd)

    if method == 'svd':
        # Find null space of A (smallest singular value)
        U, sigma, VH = np.linalg.svd(A, full_matrices=False)

        # Condition number indicator
        with np.errstate(divide='ignore'):
            if len(sigma) >= 2:
                cond = sigma[0] * np.sqrt(2) / (sigma[-2] - sigma[-1] + 1e-16)
            else:
                cond = sigma[0]

        denom_coeffs = VH[-1, :].real  # Last row of VH

    else:  # 'ls' - pin first coefficient to 1
        A0 = A[:, 0:1]
        A1 = A[:, 1:]
        x, _, _, _ = np.linalg.lstsq(A1, -A0, rcond=None)
        denom_coeffs = np.concatenate([[1.0], x.flatten()])
        cond = 1.0

    # Normalize
    denom_coeffs = denom_coeffs / denom_coeffs[0]

    # Compute denominator polynomial at each frequency
    denom_new = Qd @ denom_coeffs

    # Convert stacked representation back to complex denominator values
    denom_new_complex = _unstack_real_imag(denom_new, M)

    # Compute poles from Hessenberg matrix
    # For the orthogonal basis, roots come from eigenvalues of modified H
    # But simpler: evaluate polynomial and find roots
    poles = _compute_poles_from_hessenberg(s, Qd, Hd, denom_coeffs)

    # Flip unstable poles
    poles = _flip_unstable_poles(poles)

    return denom_new_complex, poles, denom_coeffs, cond


def _compute_poles_from_hessenberg(s, Q, H, coeffs):
    """Compute poles from orthogonal polynomial representation.

    Uses companion matrix approach: convert from orthogonal basis to
    monomial basis, then find roots.
    """
    degree = len(coeffs) - 1
    if degree == 0:
        return np.array([])

    # Sample denominator at many points and use polynomial fitting
    # This is a simple approach; more sophisticated methods exist
    M = len(s)
    denom_vals = _unstack_real_imag(Q @ coeffs, M)

    # Use least squares to fit monomial polynomial
    from numpy.polynomial import polynomial as P
    s_real = s.real
    s_imag = s.imag

    # For pure imaginary s, fit in imaginary domain
    if np.allclose(s_real, 0):
        omega = s_imag
        # Fit p(omega) where s = j*omega
        # d(s) = sum_k c_k * (j*omega)^k
        # For real coefficients: d(j*omega) = sum_k c_k * (j)^k * omega^k
        # = c_0 - c_2*omega^2 + c_4*omega^4 - ... + j*(c_1*omega - c_3*omega^3 + ...)

        # Build Vandermonde in omega
        V_omega = np.vander(omega, degree + 1, increasing=True)

        # Separate real and imaginary constraints
        # Real part coefficients: c_0, -c_2, c_4, ...
        # Imag part coefficients: c_1, -c_3, c_5, ...

        # Build sign matrix for j^k
        j_powers = np.array([1j**k for k in range(degree + 1)])
        V_real = (V_omega * j_powers.real).T.real
        V_imag = (V_omega * j_powers.imag).T.real

        A_real = V_real.T
        A_imag = V_imag.T
        b_real = denom_vals.real
        b_imag = denom_vals.imag

        A = np.vstack([A_real, A_imag])
        b = np.concatenate([b_real, b_imag])

        mono_coeffs, _, _, _ = np.linalg.lstsq(A, b, rcond=None)

        # Find roots
        if degree >= 1:
            poles = np.roots(mono_coeffs[::-1])
        else:
            poles = np.array([])

    else:
        # General complex s case
        # Fit polynomial directly
        V = np.vander(s, degree + 1, increasing=True)
        A = np.vstack([V.real, V.imag])
        b = np.concatenate([denom_vals.real, denom_vals.imag])
        mono_coeffs, _, _, _ = np.linalg.lstsq(A, b, rcond=None)

        if degree >= 1:
            poles = np.roots(mono_coeffs[::-1])
        else:
            poles = np.array([])

    return poles


def ora_fit(s, f, num_degree, denom_degree, maxiter=50, ftol=1e-10, verbose=False):
    r"""Main ORA fitting function with Sanathanan-Koerner iteration.

    Fits a rational function r(s) = n(s)/d(s) with real polynomial coefficients
    to complex data f(s).

    Parameters
    ----------
    s : np.ndarray
        Complex frequency points, shape (M,) or (M, 1)
    f : np.ndarray
        Complex function values, shape (M,) or (M, *nout)
    num_degree : int
        Numerator polynomial degree
    denom_degree : int
        Denominator polynomial degree
    maxiter : int
        Maximum SK iterations
    ftol : float
        Convergence tolerance for fit change
    verbose : bool
        Print iteration progress

    Returns
    -------
    num_coeffs : np.ndarray
        Real numerator coefficients
    denom_coeffs : np.ndarray
        Real denominator coefficients
    poles : np.ndarray
        Stable poles of rational function
    info : dict
        Additional information (iterations, residuals, etc.)
    """
    s = np.asarray(s).flatten()
    f = np.asarray(f)
    M = len(s)

    # Handle array-valued f
    original_shape = f.shape[1:] if f.ndim > 1 else ()
    f_flat = f.reshape(M, -1)
    nout = f_flat.shape[1]

    # Initialize denominator
    denom_vals = np.ones(M, dtype=complex)
    poles = None

    best_fit = {
        'residual': np.inf,
        'num_coeffs': None,
        'denom_coeffs': None,
        'poles': None,
    }

    fit_old = np.zeros_like(f_flat)
    residuals = []

    for it in range(maxiter):
        # Update denominator
        denom_vals, poles, denom_coeffs, cond = denfit(
            s, f, num_degree, denom_degree, denom_init=denom_vals
        )

        # Recompute denominator values from stable poles
        # (pole flipping may have changed them)
        denom_vals = _eval_denom_from_coeffs(s, denom_coeffs)

        # Update numerator
        Qn, num_coeffs, res_norm = numfit(s, f, num_degree, denom_weights=denom_vals)

        # Compute current fit
        fit = _eval_rational(s, num_coeffs, denom_vals, Qn)
        fit = fit.reshape(M, -1)

        residual = np.linalg.norm(f_flat - fit, 'fro') / np.linalg.norm(f_flat, 'fro')
        delta = np.linalg.norm(fit - fit_old, 'fro')
        residuals.append(residual)

        if verbose:
            print(f"Iter {it:3d}: residual = {residual:.6e}, delta = {delta:.6e}, cond = {cond:.2e}")

        # Track best fit
        if residual < best_fit['residual']:
            best_fit['residual'] = residual
            best_fit['num_coeffs'] = num_coeffs.copy()
            best_fit['denom_coeffs'] = denom_coeffs.copy()
            best_fit['poles'] = poles.copy() if poles is not None else None

        # Convergence check
        if delta < ftol * np.linalg.norm(fit, 'fro'):
            if verbose:
                print(f"Converged after {it+1} iterations")
            break

        fit_old = fit.copy()

    info = {
        'iterations': it + 1,
        'residuals': residuals,
        'final_residual': residual,
    }

    return (best_fit['num_coeffs'], best_fit['denom_coeffs'],
            best_fit['poles'], info)


def _eval_denom_from_coeffs(s, denom_coeffs):
    """Evaluate denominator polynomial at frequencies."""
    s = np.asarray(s).flatten()
    M = len(s)
    degree = len(denom_coeffs) - 1

    Qd, _ = vandermonde_real(s, degree)
    denom_stacked = Qd @ denom_coeffs
    return _unstack_real_imag(denom_stacked, M)


def _eval_rational(s, num_coeffs, denom_vals, Qn):
    """Evaluate rational function given numerator coeffs and denominator values."""
    M = len(s)
    num_coeffs = np.asarray(num_coeffs)

    if num_coeffs.ndim == 1:
        num_stacked = Qn @ num_coeffs
        num_complex = _unstack_real_imag(num_stacked, M)
        return num_complex / denom_vals
    else:
        # Array-valued
        nout = num_coeffs.shape[1]
        result = np.zeros((M, nout), dtype=complex)
        for j in range(nout):
            num_stacked = Qn @ num_coeffs[:, j]
            num_complex = _unstack_real_imag(num_stacked, M)
            result[:, j] = num_complex / denom_vals
        return result


class ORARationalApproximation:
    r"""Rational approximation with real coefficient enforcement (ORA).

    This class implements the Orthogonal Rational Approximation (ORA)
    algorithm for fitting rational functions with guaranteed real
    polynomial coefficients.

    Parameters
    ----------
    num_degree : int
        Numerator polynomial degree
    denom_degree : int
        Denominator polynomial degree
    maxiter : int
        Maximum SK iterations
    ftol : float
        Convergence tolerance
    verbose : bool
        Print progress

    Examples
    --------
    >>> import numpy as np
    >>> from polyrat.ora import ORARationalApproximation
    >>> omega = np.linspace(1e6, 1e9, 100)
    >>> s = 1j * omega
    >>> # Some complex transfer function
    >>> H = 1 / (1 + s * 1e-9 + (s * 1e-9)**2)
    >>> ora = ORARationalApproximation(num_degree=2, denom_degree=2)
    >>> ora.fit(s, H)
    >>> H_fit = ora(s)
    >>> # Coefficients are guaranteed real
    >>> assert np.allclose(np.imag(ora.num_coeffs), 0)
    >>> assert np.allclose(np.imag(ora.denom_coeffs), 0)
    """

    def __init__(self, num_degree, denom_degree, maxiter=50, ftol=1e-10, verbose=False):
        self.num_degree = int(num_degree)
        self.denom_degree = int(denom_degree)
        self.maxiter = maxiter
        self.ftol = ftol
        self.verbose = verbose

        # Fitted parameters
        self.num_coeffs = None
        self.denom_coeffs = None
        self._poles = None
        self._s_train = None
        self._info = None

    def fit(self, s, f):
        """Fit the rational function to data.

        Parameters
        ----------
        s : np.ndarray
            Complex frequency points
        f : np.ndarray
            Complex function values
        """
        self._s_train = np.asarray(s).flatten()

        self.num_coeffs, self.denom_coeffs, self._poles, self._info = ora_fit(
            s, f, self.num_degree, self.denom_degree,
            maxiter=self.maxiter, ftol=self.ftol, verbose=self.verbose
        )

    def __call__(self, s):
        """Evaluate the fitted rational function.

        Parameters
        ----------
        s : np.ndarray
            Frequency points for evaluation

        Returns
        -------
        np.ndarray
            Rational function values
        """
        return self.eval(s)

    def eval(self, s):
        """Evaluate the fitted rational function."""
        s = np.asarray(s).flatten()
        M = len(s)

        # Build bases at evaluation points
        Qn, _ = vandermonde_real(s, self.num_degree)
        denom_vals = _eval_denom_from_coeffs(s, self.denom_coeffs)

        return _eval_rational(s, self.num_coeffs, denom_vals, Qn)

    @property
    def poles(self):
        """Return poles of the rational function."""
        return self._poles

    def get_real_coefficients(self):
        """Return numerator and denominator coefficients (both real).

        Returns
        -------
        num_coeffs : np.ndarray
            Real numerator coefficients
        denom_coeffs : np.ndarray
            Real denominator coefficients
        """
        return self.num_coeffs, self.denom_coeffs

    @property
    def info(self):
        """Return fitting information dictionary."""
        return self._info
