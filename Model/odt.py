"""
Module defines the OpticalTrap class used to compute ODT physical characteristics for a fixed wavelength and input
power. Such class should be derived to form concrete classes such as OpticalTweezer, that define a specific energy
potential shape.
"""
import numpy as np
import scipy.constants as sc
from math import pi, sqrt, factorial

from Model.gaussian_bound_states import gauss_states, approximating_pulsation


MAX_LEVELS = 40


def gouldhopper(x, y, n):
    s_max = n // 2
    poly_term = 0.0
    for s in range(0, s_max+1):
        poly_term += x**(n-2*s) * y**s / factorial(n-2*s) / factorial(s)

    return factorial(n) * poly_term


def hermite(x, n):
    return gouldhopper(2*x, -1, n)


class OpticalTrap:
    """
    Generic Optical Trap class. Used as base class for different trap configurations.
    Subclasses of this should implement the bound_levels() and bound_wavefunctions() methods.
    This class is used by SidebandCooling class for computing coupling coefficients
    """
    def __init__(self, light_beam, atom_params):
        self.light_beam = light_beam
        self.atom = atom_params

    def trap_depth(self, state, x=0.0, y=0.0, z=0.0):
        pol = self.atom['polarizabilities'][state]
        v0 = -pol * self.light_beam.intensity(x, y, z) / (2 * sc.epsilon_0 * sc.c)
        return v0

    def harmonic_pulsation(self, state, x=0.0, y=0.0, perpendicular=False):
        if perpendicular:
            prefactor = 1.0 / self.light_beam.w0
        else:
            prefactor = 2 * np.pi / self.light_beam.wlen

        return prefactor * np.sqrt(2 * abs(self.trap_depth(state, x, y)) / self.atom['m'])

    def harmonic_tau(self, state, x=0.0, y=0.0, perpendicular=False):
        return self.atom["m"] * self.harmonic_pulsation(state, x, y, perpendicular) / sc.hbar

    def lamb_dicke(self, state, x=0.0, y=0.0):
        k = 2 * pi / self.atom['wlen']
        # k = 2 * pi / self.light_beam.wlen
        return k * np.sqrt(sc.hbar / (2 * self.atom["m"] * self.harmonic_pulsation(state, x, y)))

    def bound_levels(self, state, x=0.0, y=0.0):
        pass

    def bound_wavefunctions(self, state):
        pass


class OpticalTweezer(OpticalTrap):
    """
    Class implements an approximated version of Optical tweezer potential. Bound levels of the gaussian potentials
    are computed using the WBK approximation. The corresponding wavefunctions are approximated by harmonic oscillator
    wavefunctions.
    """
    def __init__(self, light_beam, atom_params):
        super(OpticalTweezer, self).__init__(light_beam, atom_params)

    def bound_levels(self, state, x=0.0, y=0.0):
        v0 = abs(self.trap_depth(state, x, y)) / sc.k
        w0 = self.light_beam.w0
        # todo: here the number of levels being computed should not be hardcoded
        levels = gauss_states(self.atom['m'], w0, v0, n_levels=60)*2*pi - v0*sc.k/sc.hbar

        # filter small and positive values
        mask = np.logical_not((levels > 0.0) & (np.abs(levels) < 1e-3))
        return levels[mask]

    def bound_wavefunctions(self, state):
        # todo: here we use the Harmonic oscillator wavefunctions, not the proper ones
        v0 = abs(self.trap_depth(state)) / sc.k
        w0 = self.light_beam.w0 / sqrt(2)
        omega = approximating_pulsation(0.0, self.atom['m'], w0, v0)
        tau = self.atom['m'] * omega / sc.hbar

        levels = self.bound_levels(state)
        n_points = 1001
        xpartition = np.linspace(-3*w0, 3*w0, n_points)
        # compute wavefunctions
        states = np.zeros(shape=(levels.shape[0], n_points))

        for n in range(levels.shape[0]):
            psi = np.exp(-tau * xpartition**2 / 2) * hermite(sqrt(tau) * xpartition, n)
            psi *= sqrt(sqrt(tau/pi)) / sqrt(2**n * factorial(n))
            states[n, :] = psi

        return states, xpartition

    def harmonic_pulsation(self, state, x=0.0, y=0.0, perpendicular=True):
        return super(OpticalTweezer, self).harmonic_pulsation(state, x, y, perpendicular)
