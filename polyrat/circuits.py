r"""Synthetic circuit S-parameter generation for testing rational approximation.

This module provides classes for generating S-parameters from electrical circuits
using symbolic computation (sympy) followed by numerical evaluation. This enables
testing of rational approximation algorithms with known ground-truth data.

References
----------
.. [Ma22] Ma & Engin (2022), "Orthogonal rational approximation of transfer
   functions for high-frequency circuits", Int. J. Circuit Theory Appl.
"""

import numpy as np
import sympy as sp
from typing import Dict, Tuple, Optional, List, Union


# Symbolic variable for complex frequency
_s = sp.Symbol('s')


class CircuitComponent:
    """Base class for circuit components with symbolic impedance.

    Parameters
    ----------
    value : float
        Component value (Ohms for R, Henries for L, Farads for C)
    name : str, optional
        Component identifier for display
    """

    def __init__(self, value: float, name: str = None):
        self.value = value
        self.name = name

    def impedance(self, s: sp.Symbol = _s) -> sp.Expr:
        """Return symbolic impedance Z(s).

        Parameters
        ----------
        s : sympy.Symbol
            Complex frequency variable

        Returns
        -------
        sp.Expr
            Symbolic impedance expression
        """
        raise NotImplementedError

    def admittance(self, s: sp.Symbol = _s) -> sp.Expr:
        """Return symbolic admittance Y(s) = 1/Z(s)."""
        return 1 / self.impedance(s)

    def __repr__(self):
        name = self.name or self.__class__.__name__
        return f"{name}({self.value})"


class Resistor(CircuitComponent):
    """Resistor with impedance Z = R."""

    def impedance(self, s: sp.Symbol = _s) -> sp.Expr:
        return sp.Rational(self.value).limit_denominator(10**12)


class Inductor(CircuitComponent):
    """Inductor with impedance Z = s*L."""

    def impedance(self, s: sp.Symbol = _s) -> sp.Expr:
        return self.value * s


class Capacitor(CircuitComponent):
    """Capacitor with impedance Z = 1/(s*C)."""

    def impedance(self, s: sp.Symbol = _s) -> sp.Expr:
        return 1 / (self.value * s)


class TwoPortNetwork:
    """Base class for two-port networks with S-parameter generation.

    Two-port networks are characterized by their ABCD (transmission) matrix:

    .. math::
        \\begin{bmatrix} V_1 \\\\ I_1 \\end{bmatrix} =
        \\begin{bmatrix} A & B \\\\ C & D \\end{bmatrix}
        \\begin{bmatrix} V_2 \\\\ I_2 \\end{bmatrix}

    S-parameters are computed from the ABCD matrix using standard conversion formulas.

    Parameters
    ----------
    Z0 : float
        Reference impedance (typically 50 Ohms)
    """

    def __init__(self, Z0: float = 50.0):
        self.Z0 = Z0
        self._s = sp.Symbol('s')
        self._abcd_cache = None
        self._sparam_cache = None
        self._sparam_funcs = None

    def abcd_matrix(self) -> sp.Matrix:
        """Return symbolic 2x2 ABCD matrix.

        Returns
        -------
        sp.Matrix
            2x2 ABCD transmission matrix as function of s
        """
        raise NotImplementedError

    def _compute_s_parameters(self) -> Dict[str, sp.Expr]:
        """Convert ABCD matrix to S-parameters.

        Uses standard conversion formulas:
            S11 = (A + B/Z0 - C*Z0 - D) / Delta
            S12 = 2*(A*D - B*C) / Delta
            S21 = 2 / Delta
            S22 = (-A + B/Z0 - C*Z0 + D) / Delta
        where Delta = A + B/Z0 + C*Z0 + D
        """
        abcd = self.abcd_matrix()
        A, B = abcd[0, 0], abcd[0, 1]
        C, D = abcd[1, 0], abcd[1, 1]
        Z0 = self.Z0

        Delta = A + B/Z0 + C*Z0 + D

        S11 = sp.simplify((A + B/Z0 - C*Z0 - D) / Delta)
        S12 = sp.simplify(2*(A*D - B*C) / Delta)
        S21 = sp.simplify(2 / Delta)
        S22 = sp.simplify((-A + B/Z0 - C*Z0 + D) / Delta)

        return {'S11': S11, 'S12': S12, 'S21': S21, 'S22': S22}

    def s_parameters(self) -> Dict[str, sp.Expr]:
        """Return symbolic S-parameters.

        Returns
        -------
        dict
            Dictionary with keys 'S11', 'S12', 'S21', 'S22' mapping to sympy expressions
        """
        if self._sparam_cache is None:
            self._sparam_cache = self._compute_s_parameters()
        return self._sparam_cache

    def _build_eval_funcs(self):
        """Build fast numerical evaluation functions using lambdify."""
        if self._sparam_funcs is None:
            s_params = self.s_parameters()
            self._sparam_funcs = {
                key: sp.lambdify(self._s, expr, modules=['numpy'])
                for key, expr in s_params.items()
            }

    def evaluate(self, frequencies: np.ndarray,
                 s_type: str = 'jw') -> Dict[str, np.ndarray]:
        """Evaluate S-parameters at given frequencies.

        Parameters
        ----------
        frequencies : np.ndarray
            Frequency values (angular frequency omega in rad/s)
        s_type : str
            'jw' for s = j*omega (default), 's' for direct complex s values

        Returns
        -------
        dict
            Dictionary mapping 'S11', 'S12', 'S21', 'S22' to complex numpy arrays
        """
        self._build_eval_funcs()

        if s_type == 'jw':
            s_vals = 1j * np.asarray(frequencies)
        else:
            s_vals = np.asarray(frequencies)

        result = {}
        for key, func in self._sparam_funcs.items():
            result[key] = np.asarray(func(s_vals), dtype=complex)

        return result

    def get_component_values(self) -> Dict[str, float]:
        """Return dictionary of component names and values."""
        raise NotImplementedError


class SeriesImpedance(TwoPortNetwork):
    """Two-port network with a series impedance element.

    ABCD matrix for series impedance Z:
        [[1, Z],
         [0, 1]]

    Parameters
    ----------
    component : CircuitComponent
        The series element
    Z0 : float
        Reference impedance
    """

    def __init__(self, component: CircuitComponent, Z0: float = 50.0):
        super().__init__(Z0)
        self.component = component

    def abcd_matrix(self) -> sp.Matrix:
        Z = self.component.impedance(self._s)
        return sp.Matrix([[1, Z], [0, 1]])

    def get_component_values(self) -> Dict[str, float]:
        return {self.component.name or 'Z': self.component.value}


class ShuntAdmittance(TwoPortNetwork):
    """Two-port network with a shunt admittance element.

    ABCD matrix for shunt admittance Y:
        [[1, 0],
         [Y, 1]]

    Parameters
    ----------
    component : CircuitComponent
        The shunt element
    Z0 : float
        Reference impedance
    """

    def __init__(self, component: CircuitComponent, Z0: float = 50.0):
        super().__init__(Z0)
        self.component = component

    def abcd_matrix(self) -> sp.Matrix:
        Y = self.component.admittance(self._s)
        return sp.Matrix([[1, 0], [Y, 1]])

    def get_component_values(self) -> Dict[str, float]:
        return {self.component.name or 'Y': self.component.value}


class SeriesRLC(TwoPortNetwork):
    """Series RLC circuit as a two-port network.

    The series impedance is Z = R + s*L + 1/(s*C).

    Parameters
    ----------
    R : float
        Resistance in Ohms
    L : float
        Inductance in Henries
    C : float
        Capacitance in Farads
    Z0 : float
        Reference impedance
    """

    def __init__(self, R: float, L: float, C: float, Z0: float = 50.0):
        super().__init__(Z0)
        self.R = R
        self.L = L
        self.C = C

    def abcd_matrix(self) -> sp.Matrix:
        s = self._s
        Z_series = self.R + self.L * s + 1/(self.C * s)
        return sp.Matrix([[1, Z_series], [0, 1]])

    def get_component_values(self) -> Dict[str, float]:
        return {'R': self.R, 'L': self.L, 'C': self.C}


class ParallelRLC(TwoPortNetwork):
    """Parallel RLC circuit as a two-port shunt network.

    The shunt admittance is Y = 1/R + 1/(s*L) + s*C.

    Parameters
    ----------
    R : float
        Resistance in Ohms
    L : float
        Inductance in Henries
    C : float
        Capacitance in Farads
    Z0 : float
        Reference impedance
    """

    def __init__(self, R: float, L: float, C: float, Z0: float = 50.0):
        super().__init__(Z0)
        self.R = R
        self.L = L
        self.C = C

    def abcd_matrix(self) -> sp.Matrix:
        s = self._s
        Y_shunt = 1/self.R + 1/(self.L * s) + self.C * s
        return sp.Matrix([[1, 0], [Y_shunt, 1]])

    def get_component_values(self) -> Dict[str, float]:
        return {'R': self.R, 'L': self.L, 'C': self.C}


class CascadedNetwork(TwoPortNetwork):
    """Cascaded (series-connected) two-port networks.

    The overall ABCD matrix is the product of individual ABCD matrices:
        ABCD_total = ABCD_1 * ABCD_2 * ... * ABCD_n

    Parameters
    ----------
    networks : list of TwoPortNetwork
        Networks to cascade (in order from input to output)
    Z0 : float
        Reference impedance
    """

    def __init__(self, networks: List[TwoPortNetwork], Z0: float = 50.0):
        super().__init__(Z0)
        self.networks = networks

    def abcd_matrix(self) -> sp.Matrix:
        result = sp.eye(2)
        for network in self.networks:
            # Ensure consistent s variable
            network._s = self._s
            network._sparam_cache = None  # Clear cache
            result = result * network.abcd_matrix()
        return result

    def get_component_values(self) -> Dict[str, float]:
        values = {}
        for i, net in enumerate(self.networks):
            for key, val in net.get_component_values().items():
                values[f"{key}_{i}"] = val
        return values


class LadderNetwork(TwoPortNetwork):
    """Ladder network with alternating series and shunt elements.

    Topology: Series -> Shunt -> Series -> Shunt -> ...

    Parameters
    ----------
    series_components : list of CircuitComponent
        Components for series branches
    shunt_components : list of CircuitComponent
        Components for shunt branches
    Z0 : float
        Reference impedance
    """

    def __init__(self,
                 series_components: List[CircuitComponent],
                 shunt_components: List[CircuitComponent],
                 Z0: float = 50.0):
        super().__init__(Z0)
        self.series_components = series_components
        self.shunt_components = shunt_components

    def abcd_matrix(self) -> sp.Matrix:
        s = self._s
        result = sp.eye(2)

        n_series = len(self.series_components)
        n_shunt = len(self.shunt_components)

        for i in range(max(n_series, n_shunt)):
            if i < n_series:
                Z = self.series_components[i].impedance(s)
                series_abcd = sp.Matrix([[1, Z], [0, 1]])
                result = result * series_abcd

            if i < n_shunt:
                Y = self.shunt_components[i].admittance(s)
                shunt_abcd = sp.Matrix([[1, 0], [Y, 1]])
                result = result * shunt_abcd

        return result

    def get_component_values(self) -> Dict[str, float]:
        values = {}
        for i, comp in enumerate(self.series_components):
            values[f"series_{i}"] = comp.value
        for i, comp in enumerate(self.shunt_components):
            values[f"shunt_{i}"] = comp.value
        return values


class LowpassLadder(TwoPortNetwork):
    """Standard lowpass ladder filter (Butterworth prototype).

    Generates component values for Butterworth response.

    Parameters
    ----------
    order : int
        Filter order (1 or 2)
    cutoff_freq : float
        Cutoff frequency in rad/s
    Z0 : float
        Reference impedance
    """

    def __init__(self, order: int, cutoff_freq: float, Z0: float = 50.0):
        super().__init__(Z0)
        assert order in [1, 2], "Only order 1 and 2 are supported"
        self.order = order
        self.cutoff_freq = cutoff_freq
        self._compute_components()

    def _compute_components(self):
        """Compute Butterworth prototype component values."""
        # Butterworth g-values (normalized)
        if self.order == 1:
            g = [1.0, 2.0, 1.0]
        else:  # order == 2
            g = [1.0, 1.4142, 1.4142, 1.0]

        self.components = []
        for k in range(1, self.order + 1):
            if k % 2 == 1:  # Odd index: series inductor
                L = g[k] * self.Z0 / self.cutoff_freq
                self.components.append(('L', Inductor(L, f'L{k}')))
            else:  # Even index: shunt capacitor
                C = g[k] / (self.Z0 * self.cutoff_freq)
                self.components.append(('C', Capacitor(C, f'C{k}')))

        # Add terminating shunt capacitor for odd-order
        if self.order % 2 == 1:
            C = g[self.order + 1] / (self.Z0 * self.cutoff_freq)
            self.components.append(('C', Capacitor(C, f'C{self.order+1}')))

    def abcd_matrix(self) -> sp.Matrix:
        s = self._s
        result = sp.eye(2)

        for comp_type, component in self.components:
            if comp_type == 'L':
                Z = component.impedance(s)
                abcd = sp.Matrix([[1, Z], [0, 1]])
            else:  # 'C'
                Y = component.admittance(s)
                abcd = sp.Matrix([[1, 0], [Y, 1]])
            result = result * abcd

        return result

    def get_component_values(self) -> Dict[str, float]:
        return {comp.name: comp.value for _, comp in self.components}


# Convenience functions for generating test data

def generate_sparam_data(circuit: TwoPortNetwork,
                         freq_start: float,
                         freq_stop: float,
                         n_points: int,
                         log_scale: bool = True) -> Tuple[np.ndarray, np.ndarray]:
    """Generate S-parameter test data from a circuit.

    Parameters
    ----------
    circuit : TwoPortNetwork
        The circuit to evaluate
    freq_start, freq_stop : float
        Frequency range in rad/s
    n_points : int
        Number of frequency points
    log_scale : bool
        If True, use logarithmically spaced frequencies

    Returns
    -------
    X : np.ndarray, shape (n_points, 1)
        Complex frequency values s = j*omega
    Y : np.ndarray, shape (n_points, 2, 2)
        S-parameter matrix at each frequency
    """
    if log_scale:
        freqs = np.logspace(np.log10(freq_start), np.log10(freq_stop), n_points)
    else:
        freqs = np.linspace(freq_start, freq_stop, n_points)

    s_vals = 1j * freqs
    X = s_vals.reshape(-1, 1)

    s_params = circuit.evaluate(freqs, s_type='jw')

    Y = np.zeros((n_points, 2, 2), dtype=complex)
    Y[:, 0, 0] = s_params['S11']
    Y[:, 0, 1] = s_params['S12']
    Y[:, 1, 0] = s_params['S21']
    Y[:, 1, 1] = s_params['S22']

    return X, Y


def generate_reflection_data(circuit: TwoPortNetwork,
                             freq_start: float,
                             freq_stop: float,
                             n_points: int,
                             log_scale: bool = True) -> Tuple[np.ndarray, np.ndarray]:
    """Generate S11 (reflection coefficient) test data.

    Parameters
    ----------
    circuit : TwoPortNetwork
        The circuit to evaluate
    freq_start, freq_stop : float
        Frequency range in rad/s
    n_points : int
        Number of frequency points
    log_scale : bool
        If True, use logarithmically spaced frequencies

    Returns
    -------
    X : np.ndarray, shape (n_points, 1)
        Complex frequency values s = j*omega
    y : np.ndarray, shape (n_points,)
        S11 values
    """
    X, Y = generate_sparam_data(circuit, freq_start, freq_stop, n_points, log_scale)
    return X, Y[:, 0, 0]
