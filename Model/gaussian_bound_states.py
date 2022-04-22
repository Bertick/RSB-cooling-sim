import numpy as np
import scipy.optimize as opt
import scipy.constants as sc

from math import pi, sqrt

hbar = sc.hbar
kb = sc.k


def potential(x, w0, v0):
    return v0 * kb * (1 - np.exp(-2 * (x/w0)**2))


def potential_p(x, w0, v0):
    return 2 * v0 * kb * (2/w0**2) * x * np.exp(-2 * (x/w0)**2)


def potential_pp(x, w0, v0):
    return -2 * v0 * kb * (2/w0**2) * np.exp(-2 * (x/w0)**2) * (4/w0**2 * x**2 - 1)


def approximating_pulsation(x0, m, w0, v0):
    return sqrt(potential_pp(x0, w0, v0) / m)


def approximating_parabola(x, e, w0, v0):
    # compute the classical inversion point at the given energy
    r0 = inv_point(e, w0, v0)
    # compute the 2nd-order Taylor series around r0
    return potential(r0, w0, v0) + potential_p(r0, w0, v0) * (x-r0) + 0.5 * potential_pp(r0, w0, v0) * (x-r0)**2


def inv_point(e, w0, v0):
    return np.sqrt(-(w0**2/2) * np.log(1 - e*2*pi*hbar / v0/kb))


def bkw_condition(e, n, m, w0, v0, *, n_points=10001):
    """See: D. Comparat, arxiv:0305157"""
    # compute classical inversion points
    r1 = inv_point(e, w0, v0)
    r0 = -r1

    # generate space partition
    r_space = np.linspace(r0, r1, n_points)

    # dx = (r1-r0)/n_points
    dx = r_space[1] - r_space[0]
    # compute integral
    integrand = e*2*pi*hbar - potential(r_space, w0, v0)
    # the extrema may become negative (subtraction of very similar numbers)
    integrand[0] = 0.0
    integrand[-1] = 0.0
    # integral
    ii = np.sum(np.sqrt(integrand) * dx)
    ii *= sqrt(2*m) / pi / hbar
    # return the condition to be set to zero
    return ii - (n+1/2)


def gauss_states(m, w0, v0, n_levels=100):
    eigvals = [0.0] * n_levels

    w = approximating_pulsation(0.0, m, w0, v0)

    harmonic_levels = (np.arange(2*n_levels) + 0.5) * w / 2 / pi

    for n in range(n_levels):

        n0 = n-4 if n > 4 else 0
        n1 = n+4

        e0 = eigvals[n-1] if n > 0 else 0.0

        sol = opt.root_scalar(
            bkw_condition,
            args=(n, m, w0, v0),
            bracket=(e0, harmonic_levels[n1]),
            method='bisect',
            rtol=1e-12,
            xtol=1e-9
        )
        eigvals[n] = sol.root

    eigvals = np.array(eigvals)

    # clear states that are either 0.0 or very close to v0
    mask = np.logical_not((eigvals < 1e-6) & (eigvals - v0*sc.k/2/pi/hbar < 1e-6))

    return eigvals[mask]


if __name__ == '__main__':
    w0 = 0.5e-6
    v0 = 1e-4
    m0 = 10 * sc.u

    for x in np.array([1.0, 2.5, 5.0, 7.5, 10.0]):
        m = m0 * x
        v = v0 * x
        lvls = gauss_states(m, w0, v, n_levels=10)
        print(lvls)
        print()