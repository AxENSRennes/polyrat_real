"""Tests for Orthogonal Rational Approximation (ORA) with real coefficients."""
import numpy as np
import pytest
from polyrat.ora import (
    ORARationalApproximation,
    ora_fit, numfit, denfit,
    vandermonde_real, _stack_real_imag, _unstack_real_imag
)
from polyrat.arnoldi_real import lanczos_real, RealPolynomialBasis


class TestStackUnstack:
    """Test stacking/unstacking of real/imaginary parts."""

    def test_roundtrip(self):
        z = np.array([1+2j, 3-4j, 5+0j, 0-6j])
        stacked = _stack_real_imag(z)
        unstacked = _unstack_real_imag(stacked, len(z))
        np.testing.assert_allclose(unstacked, z)

    def test_shapes(self):
        z = np.random.randn(100) + 1j * np.random.randn(100)
        stacked = _stack_real_imag(z)
        assert stacked.shape == (200,)
        assert np.isrealobj(stacked)


class TestLanczosReal:
    """Test real-valued Lanczos/Arnoldi iteration."""

    def test_orthogonality(self):
        omega = np.linspace(1e6, 1e9, 100)
        s = 1j * omega

        Q, H = lanczos_real(s, degree=5)

        # Q should be orthonormal
        QtQ = Q.T @ Q
        np.testing.assert_allclose(QtQ, np.eye(Q.shape[1]), atol=1e-10)

    def test_real_output(self):
        omega = np.linspace(1e6, 1e9, 100)
        s = 1j * omega

        Q, H = lanczos_real(s, degree=5)

        assert np.isrealobj(Q), "Q should be real"
        assert np.isrealobj(H), "H should be real"


class TestRealCoefficientEnforcement:
    """Test that coefficients are indeed real."""

    def test_numfit_real_coeffs(self):
        omega = np.linspace(1e6, 1e9, 50)
        s = 1j * omega

        # Create test data from known real-coefficient polynomial
        # p(s) = 1 + 2*s + 3*s^2 (real coeffs)
        f = 1 + 2*s + 3*s**2

        Q, coeffs, res = numfit(s, f, num_degree=2)

        # Coefficients should be real
        assert np.isrealobj(coeffs), "Numerator coefficients should be real"

    def test_ora_real_coeffs(self):
        omega = np.logspace(6, 9, 100)
        s = 1j * omega

        # Create test data from known real-coefficient rational
        # H(s) = (1 + s*1e-9) / (1 + 2*s*1e-9 + (s*1e-9)^2)
        tau = 1e-9
        num = 1 + s * tau
        denom = 1 + 2*s*tau + (s*tau)**2
        H = num / denom

        ora = ORARationalApproximation(num_degree=1, denom_degree=2, maxiter=20, verbose=False)
        ora.fit(s, H)

        num_c, denom_c = ora.get_real_coefficients()

        assert np.isrealobj(num_c), "Numerator coefficients should be real"
        assert np.isrealobj(denom_c), "Denominator coefficients should be real"


class TestFittingQuality:
    """Test that ORA achieves good fitting quality."""

    def test_simple_lowpass(self):
        """Test fitting a simple first-order lowpass."""
        omega = np.logspace(6, 10, 100)
        s = 1j * omega

        tau = 1e-9
        H = 1 / (1 + s * tau)

        ora = ORARationalApproximation(num_degree=0, denom_degree=1, maxiter=30)
        ora.fit(s, H)
        H_fit = ora(s)

        rel_err = np.linalg.norm(H - H_fit) / np.linalg.norm(H)
        assert rel_err < 1e-4, f"Relative error {rel_err} too large"

    def test_second_order_system(self):
        """Test fitting a second-order system."""
        omega = np.logspace(6, 10, 100)
        s = 1j * omega

        # Second order lowpass with resonance
        wn = 1e9  # natural frequency
        zeta = 0.3  # damping ratio
        H = wn**2 / (s**2 + 2*zeta*wn*s + wn**2)

        ora = ORARationalApproximation(num_degree=0, denom_degree=2, maxiter=50)
        ora.fit(s, H)
        H_fit = ora(s)

        rel_err = np.linalg.norm(H - H_fit) / np.linalg.norm(H)
        assert rel_err < 1e-3, f"Relative error {rel_err} too large"


class TestStability:
    """Test pole stability enforcement."""

    def test_poles_stable(self):
        """Verify all poles have non-positive real parts."""
        omega = np.logspace(6, 10, 100)
        s = 1j * omega

        wn = 1e9
        zeta = 0.3
        H = wn**2 / (s**2 + 2*zeta*wn*s + wn**2)

        ora = ORARationalApproximation(num_degree=0, denom_degree=2, maxiter=30)
        ora.fit(s, H)

        poles = ora.poles
        if poles is not None and len(poles) > 0:
            assert np.all(poles.real <= 0), "All poles should have Re(p) <= 0"


class TestArrayValued:
    """Test array-valued (multi-output) fitting."""

    def test_multioutput_shape(self):
        omega = np.logspace(6, 9, 50)
        s = 1j * omega
        M = len(s)

        # Two-output system
        H = np.zeros((M, 2), dtype=complex)
        tau1, tau2 = 1e-9, 2e-9
        H[:, 0] = 1 / (1 + s * tau1)
        H[:, 1] = 1 / (1 + s * tau2)

        ora = ORARationalApproximation(num_degree=0, denom_degree=1)
        ora.fit(s, H)
        H_fit = ora(s)

        # Check shape
        assert H_fit.shape == (M,) or H_fit.shape == (M, 2)

    def test_multioutput_joint_denominator_fit(self):
        """Regression test: verify denominator is fit using ALL outputs jointly.

        This test creates multiple outputs sharing the SAME denominator but with
        different numerators. The buggy code would fit the denominator using only
        the last output, resulting in poor fits for earlier outputs.
        """
        omega = np.logspace(6, 10, 100)
        s = 1j * omega
        M = len(s)

        # Create 3 outputs with SAME denominator but different numerators
        # H_k(s) = (a_k + b_k*s) / (1 + 2*s*tau + (s*tau)^2)
        tau = 1e-9
        denom = 1 + 2*s*tau + (s*tau)**2

        H = np.zeros((M, 3), dtype=complex)
        H[:, 0] = (1.0 + 0.5*s*tau) / denom  # First output
        H[:, 1] = (0.8 - 0.3*s*tau) / denom  # Second output
        H[:, 2] = (0.5 + 0.2*s*tau) / denom  # Third output

        ora = ORARationalApproximation(num_degree=1, denom_degree=2, maxiter=30, verbose=False)
        ora.fit(s, H)
        H_fit = ora(s)

        # Compute relative error for each output
        rel_err = np.zeros(3)
        for j in range(3):
            rel_err[j] = np.linalg.norm(H[:, j] - H_fit[:, j]) / np.linalg.norm(H[:, j])

        # All outputs should be fit well (not just the last one!)
        for j in range(3):
            assert rel_err[j] < 0.01, f"Output {j} has rel error {rel_err[j]:.4f}, expected < 0.01"

        # The errors should be comparable (not first >> last)
        # This would fail with the buggy code where only last output was used for denominator
        assert rel_err[0] < 10 * rel_err[2], \
            f"First output error {rel_err[0]:.4f} >> last {rel_err[2]:.4f}; denominator may not use all outputs"


class TestIntegrationWithCircuits:
    """Integration tests with circuit-generated data."""

    def test_fit_series_rlc(self):
        """Fit S11 from a series RLC circuit."""
        try:
            from polyrat.circuits import SeriesRLC, generate_reflection_data
        except ImportError:
            pytest.skip("circuits module not available")

        circuit = SeriesRLC(R=10.0, L=1e-6, C=1e-9, Z0=50.0)
        X, y = generate_reflection_data(circuit, 1e6, 1e10, 100)
        s = X.flatten()

        ora = ORARationalApproximation(num_degree=2, denom_degree=2, maxiter=50, verbose=False)
        ora.fit(s, y)
        y_fit = ora(s)

        rel_err = np.linalg.norm(y - y_fit) / np.linalg.norm(y)
        assert rel_err < 0.05, f"Relative error {rel_err} too large for circuit fit"

        # Verify real coefficients
        num_c, denom_c = ora.get_real_coefficients()
        assert np.allclose(np.imag(num_c), 0, atol=1e-10)
        assert np.allclose(np.imag(denom_c), 0, atol=1e-10)


class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_degree_zero(self):
        """Test with zero-degree numerator (constant)."""
        omega = np.linspace(1e6, 1e9, 50)
        s = 1j * omega

        H = np.ones_like(s) * 0.5  # Constant

        ora = ORARationalApproximation(num_degree=0, denom_degree=0)
        ora.fit(s, H)
        H_fit = ora(s)

        np.testing.assert_allclose(np.abs(H_fit), 0.5, rtol=0.1)

    def test_few_points(self):
        """Test with minimum number of frequency points."""
        omega = np.array([1e6, 1e7, 1e8, 1e9])
        s = 1j * omega

        H = 1 / (1 + s * 1e-9)

        ora = ORARationalApproximation(num_degree=0, denom_degree=1)
        ora.fit(s, H)

        # Should not raise


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
