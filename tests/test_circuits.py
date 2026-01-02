"""Tests for synthetic circuit S-parameter generation."""
import numpy as np
import pytest
from polyrat.circuits import (
    Resistor, Inductor, Capacitor,
    SeriesImpedance, ShuntAdmittance,
    SeriesRLC, ParallelRLC,
    LadderNetwork, LowpassLadder, CascadedNetwork,
    generate_sparam_data, generate_reflection_data
)


class TestCircuitComponents:
    """Test individual circuit components."""

    def test_resistor_impedance(self):
        R = Resistor(100.0)
        import sympy as sp
        s = sp.Symbol('s')
        Z = R.impedance(s)
        # Resistor impedance should be constant (no s dependence)
        assert Z.is_constant()
        assert float(Z) == 100.0

    def test_inductor_impedance(self):
        L = Inductor(1e-6)  # 1 uH
        import sympy as sp
        s = sp.Symbol('s')
        Z = L.impedance(s)
        # Z = s*L
        assert Z.coeff(s) == 1e-6

    def test_capacitor_impedance(self):
        C = Capacitor(1e-9)  # 1 nF
        import sympy as sp
        s = sp.Symbol('s')
        Z = C.impedance(s)
        # Z = 1/(s*C)
        result = sp.simplify(Z * s * 1e-9)
        assert abs(float(result.evalf()) - 1.0) < 1e-10


class TestPassivity:
    """Test that all S-parameter magnitudes are <= 1 for passive networks."""

    @pytest.mark.parametrize("R,L,C", [
        (50.0, 1e-6, 1e-9),
        (100.0, 1e-9, 1e-12),
        (10.0, 1e-3, 1e-6),
    ])
    def test_series_rlc_passivity(self, R, L, C):
        circuit = SeriesRLC(R, L, C)
        freqs = np.logspace(6, 12, 100)  # 1 MHz to 1 THz
        s_params = circuit.evaluate(freqs)

        for key in ['S11', 'S12', 'S21', 'S22']:
            mag = np.abs(s_params[key])
            assert np.all(mag <= 1.0 + 1e-10), f"|{key}| > 1 for SeriesRLC"

    @pytest.mark.parametrize("R,L,C", [
        (50.0, 1e-6, 1e-9),
        (100.0, 1e-9, 1e-12),
        (10.0, 1e-3, 1e-6),
    ])
    def test_parallel_rlc_passivity(self, R, L, C):
        circuit = ParallelRLC(R, L, C)
        freqs = np.logspace(6, 12, 100)
        s_params = circuit.evaluate(freqs)

        for key in ['S11', 'S12', 'S21', 'S22']:
            mag = np.abs(s_params[key])
            assert np.all(mag <= 1.0 + 1e-10), f"|{key}| > 1 for ParallelRLC"

    def test_lowpass_ladder_passivity(self):
        for order in [1, 2]:
            circuit = LowpassLadder(order, cutoff_freq=1e9)
            freqs = np.logspace(6, 12, 100)
            s_params = circuit.evaluate(freqs)

            for key in ['S11', 'S12', 'S21', 'S22']:
                mag = np.abs(s_params[key])
                assert np.all(mag <= 1.0 + 1e-10), f"|{key}| > 1 for order {order}"


class TestReciprocity:
    """Test that S12 = S21 for reciprocal networks."""

    def test_series_rlc_reciprocity(self):
        circuit = SeriesRLC(50.0, 1e-6, 1e-9)
        freqs = np.logspace(6, 12, 100)
        s_params = circuit.evaluate(freqs)

        np.testing.assert_allclose(
            s_params['S12'], s_params['S21'],
            rtol=1e-10, atol=1e-14,
            err_msg="S12 != S21 for SeriesRLC"
        )

    def test_parallel_rlc_reciprocity(self):
        circuit = ParallelRLC(50.0, 1e-6, 1e-9)
        freqs = np.logspace(6, 12, 100)
        s_params = circuit.evaluate(freqs)

        np.testing.assert_allclose(
            s_params['S12'], s_params['S21'],
            rtol=1e-10, atol=1e-14,
            err_msg="S12 != S21 for ParallelRLC"
        )

    def test_ladder_reciprocity(self):
        series_comps = [Inductor(1e-6), Inductor(2e-6)]
        shunt_comps = [Capacitor(1e-9), Capacitor(2e-9)]
        circuit = LadderNetwork(series_comps, shunt_comps)

        freqs = np.logspace(6, 12, 100)
        s_params = circuit.evaluate(freqs)

        np.testing.assert_allclose(
            s_params['S12'], s_params['S21'],
            rtol=1e-10, atol=1e-14,
            err_msg="S12 != S21 for LadderNetwork"
        )


class TestKnownValues:
    """Test S-parameters against known analytical values."""

    def test_matched_load(self):
        """A series 50 Ohm resistor should give S11 = 1/3, S21 = 2/3 at DC limit."""
        R = Resistor(50.0)
        circuit = SeriesImpedance(R, Z0=50.0)

        # Use very low frequency (near DC)
        freqs = np.array([1e-3])
        s_params = circuit.evaluate(freqs)

        # For series 50 Ohm with Z0=50: S11 = Z/(2*Z0 + Z) = 50/150 = 1/3
        # S21 = 2*Z0/(2*Z0 + Z) = 100/150 = 2/3
        np.testing.assert_allclose(
            np.atleast_1d(s_params['S11'])[0], 1/3, rtol=1e-8,
            err_msg="S11 for series R != expected"
        )
        np.testing.assert_allclose(
            np.atleast_1d(s_params['S21'])[0], 2/3, rtol=1e-8,
            err_msg="S21 for series R != expected"
        )

    def test_through_connection(self):
        """Zero impedance should give S11 = 0, S21 = 1."""
        R = Resistor(0.0)
        circuit = SeriesImpedance(R, Z0=50.0)

        freqs = np.array([1e9])
        s_params = circuit.evaluate(freqs)

        np.testing.assert_allclose(
            np.atleast_1d(s_params['S11'])[0], 0.0, atol=1e-10,
            err_msg="S11 for through != 0"
        )
        np.testing.assert_allclose(
            np.atleast_1d(s_params['S21'])[0], 1.0, atol=1e-10,
            err_msg="S21 for through != 1"
        )


class TestDataGeneration:
    """Test convenience data generation functions."""

    def test_generate_sparam_data_shape(self):
        circuit = SeriesRLC(50.0, 1e-6, 1e-9)
        X, Y = generate_sparam_data(circuit, 1e6, 1e9, 50)

        assert X.shape == (50, 1)
        assert Y.shape == (50, 2, 2)
        assert np.iscomplexobj(X)
        assert np.iscomplexobj(Y)

    def test_generate_reflection_data_shape(self):
        circuit = SeriesRLC(50.0, 1e-6, 1e-9)
        X, y = generate_reflection_data(circuit, 1e6, 1e9, 50)

        assert X.shape == (50, 1)
        assert y.shape == (50,)
        assert np.iscomplexobj(X)
        assert np.iscomplexobj(y)

    def test_log_vs_linear_scale(self):
        circuit = SeriesRLC(50.0, 1e-6, 1e-9)

        X_log, _ = generate_sparam_data(circuit, 1e6, 1e9, 50, log_scale=True)
        X_lin, _ = generate_sparam_data(circuit, 1e6, 1e9, 50, log_scale=False)

        # Log scale should have larger ratios between consecutive points
        ratios_log = np.abs(X_log[1:, 0]) / np.abs(X_log[:-1, 0])
        ratios_lin = np.abs(X_lin[1:, 0]) / np.abs(X_lin[:-1, 0])

        # Log ratios should be nearly constant
        assert np.std(ratios_log) < np.std(ratios_lin)


class TestCascadedNetwork:
    """Test cascaded two-port networks."""

    def test_cascade_order_matters(self):
        """Verify that cascading order can affect result for asymmetric networks."""
        L = SeriesImpedance(Inductor(1e-6))
        C = ShuntAdmittance(Capacitor(1e-9))

        cascade1 = CascadedNetwork([L, C])
        cascade2 = CascadedNetwork([C, L])

        freqs = np.logspace(6, 10, 50)
        s1 = cascade1.evaluate(freqs)
        s2 = cascade2.evaluate(freqs)

        # S21 should be the same (reciprocal), but S11 and S22 may differ
        np.testing.assert_allclose(s1['S21'], s2['S21'], rtol=1e-10)
        # S11 and S22 are swapped between the two orderings
        np.testing.assert_allclose(s1['S11'], s2['S22'], rtol=1e-10)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
