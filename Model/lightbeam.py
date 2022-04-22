"""
Module implements the GaussianBeam class which represents a gaussian-like laser beam.
The scattering rate expression is taken from Stenholm1986 (10.1103/RevModPhys.58.699).
All other formulae are taken from Yariv's "optical electronics in modern communications".
"""

import numpy as np
from scipy.special import eval_hermite
import scipy.constants as sc

from math import sqrt, pi


class GaussianBeam:
    def __init__(self, wlen, w0=None, z0=None, i0=1.0, is_standing=False):

        # either w0 or zo must be specified
        if w0 is None and z0 is None:
            raise ValueError("You must specify either the waist (w0) or rayleigh range (z0) of the beam")

        assert i0 > 0.0

        # define attributes
        self._wlen = wlen
        self._i0 = i0

        if w0 is None:
            self._z0 = z0
            self._w0 = sqrt(wlen * z0 / pi)
        else:
            self._w0 = w0
            self._z0 = pi * w0**2 / wlen

        self.is_standing = is_standing

    # waist, rayleigh range and wlen are protected, since everything needs recalculation if they change
    @property
    def w0(self):
        return self._w0

    @w0.setter
    def w0(self, val):
        assert val > 0
        self._w0 = val
        self._z0 = pi * val**2 / self._wlen

    @property
    def z0(self):
        return self._z0

    @z0.setter
    def z0(self, val):
        assert val > 0.0
        self._z0 = val
        self._w0 = sqrt(self._wlen * val / pi)

    @property
    def wlen(self):
        return self._wlen

    @wlen.setter
    def wlen(self, val):
        assert val > 0.0
        self._wlen = val
        self._z0 = pi * self._w0**2 / val

    @property
    def i0(self):
        return self._i0

    @i0.setter
    def i0(self, val):
        assert val > 0.0
        self._i0 = val

    #############################
    # DEPENDENT BEAM ATTRIBUTES #
    #############################
    def waist(self, z=0):
        return self._w0 * np.sqrt(1 + (z/self._z0)**2)

    def raleigh_r(self, z=0):
        return z*(1 + (self._z0/z)**2)

    def raleigh_q(self, z):
        if z == 0:
            return (- 1j * (self._wlen / (np.pi * self.waist(z) ** 2))) ** (-1)
        return (1 / self.raleigh_r(z) - 1j*(self._wlen / (np.pi * self.waist(z)**2)))**(-1)

    def gouy(self, z):
        return np.arctan(z / self._z0)

    # Hermite-Gauss beam amplitude
    def hermite_amp(self, x, y, z, l, m, t):
        wz = self.waist(z)
        k = 2 * pi / self.wlen

        hermx = eval_hermite(l, np.sqrt(2) * x / wz) * np.exp(-(x / wz)**2)
        hermy = eval_hermite(m, np.sqrt(2) * y / wz) * np.exp(-(y / wz)**2)

        if z != 0:
            phase = z*k - t*k*sc.c - k*(x**2 + y**2)/(2*self.raleigh_r(z)) - (l+m+1)*self.gouy(z)
        else:
            phase = t*k*sc.c

        return (self._w0/wz) * hermx * hermy * np.exp(1j*phase) * sqrt(self._i0)

    def standingwave_a(self, x, y, z, l=0, m=0, t=0):
        return self.hermite_amp(x, y, 0, l, m, t) * 2 * np.cos(2*pi*z/self._wlen)

    def standingwave_i(self, x, y, z, l=0, m=0, t=0):
        return np.abs(self.standingwave_a(x, y, z, l, m, t))**2

    def intensity(self, x, y, z, l=0, m=0, t=0):
        if self.is_standing:
            return self.standingwave_i(x, y, z, l, m, t)
        else:
            return np.abs(self.hermite_amp(x, y, z, l, m, t))**2
