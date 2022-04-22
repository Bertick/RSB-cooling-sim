"""
Module implements the formulae for converting dipole elements between different quantum states basis. Formulae
follow the notation in M. E. Gehm. “Preparation of an Optically-Trapped Degenerate Fermi Gas of 6Li:
Finding theRoute to Degeneracy”. PhD thesis. Duke University, 2003.
"""
import numpy as np
from sympy.physics.wigner import wigner_3j, wigner_6j
from functools import lru_cache


@lru_cache(maxsize=128)
def get_wigner_3j(j0, j1, j2, m0, m1, m2):
    return float(wigner_3j(j0, j1, j2, m0, m1, m2))


@lru_cache(maxsize=32)
def get_wigner_6j(j0, j1, j2, j3, j4, j5):
    return float(wigner_6j(j0, j1, j2, j3, j4, j5))


# force pre-cache of wigner 3j and 6j for common Lithium values
[get_wigner_3j(j1, 1, j0, -mj1, mj1-mj0, mj0)
 for j0 in [0.5, 1.5]
 for j1 in [0.5, 1.5]
 for mj1 in np.arange(-1.5, 1.5, 1)
 for mj0 in np.arange(-1.5, 1.5, 1)]

[get_wigner_6j(0.5, 1.0, f1, f0, 1, 0.5)
 for f0 in [0.5, 1.5]
 for f1 in [0.5, 1.5]]


def gamma_hfs(gamma_fs, j0, f0, mf0, j1, f1, mf1, ii, q=None):
    """
    Computes the hyperfine transition rate gamma from the fine structure transition rate.
    The same expression can be used for computing the fine structure gamma by making the substitution F->J, mF->mJ,
    ii->S and gamma_fs->gamma.
    :param gamma_fs: Decay rate of the fine structure transition (5.8MHz for Lithium)
    :param j0: initial state angular momentum quantum number
    :param f0: initial state angular momentum quantum number
    :param mf0: initial state magnetic quantum number
    :param j1: final state angular momentum quantum number
    :param f1: final state angular momentum quantum number
    :param mf1: final state magnetic quantum number
    :param ii: nuclear spin
    :param q: Light polarization (defaults to pi-polarization q=0)
    :return: decay rate of hyperfine transition
    """
    if q is None:
        q = mf1 - mf0
    return 2 * gamma_fs * (reduced2dipel(1.0, f0, mf0, f1, mf1, q) * redipel_hfs(1.0, j0, j1, f0, f1, ii)) ** 2


def reduced2dipel(redipel_fs, j0, mj0, j1, mj1, q):
    """
    Computes the dipole element <J, mJ|dq|J',mJ'> from the reduced dipole matrix element <J|\vec{d}|J'>.
    The same expression can be used for hyperfine levels by making the substitutions J->F, mJ->mF.
    :param redipel_fs: reduced dipole matrix element <J|\vec{d}|J'>
    :param j0: initial state angular momentum quantum number
    :param mj0: initial state magnetic quantum number
    :param j1: final state angular momentum quantum number
    :param mj1: final state magnetic quantum number
    :param q: dipole element component in spherical coordinates (either 0, 1 or -1)
    :return: non-reduced dipole matrix element
    """
    return redipel_fs * (-1)**(j1 - mj1) * get_wigner_3j(j1, 1, j0, -mj1, q, mj0)


def redipel_hfs(redipel_fs, j0, j1, f0, f1, ii):
    """
    Computes the hyperfine-structure reduced dipole element starting from the fine-structure reduced dipole element.
    The same expression can be used for fine-structure by making the substitution J->L, I->S, F->J.
    :param redipel_fs: reduced dipole matrix element <J|\vec{d}|J'>
    :param j0: initial state angular momentum quantum number
    :param j1: final state angular momentum quantum number
    :param f0: initial state angular momentum quantum number
    :param f1: final state angular momentum quantum number
    :param ii: nuclear spin quantum number
    :return: hyperfine-structure reduced dipole matrix element
    """
    return redipel_fs * (-1)**(j1+f0+1+ii) * np.sqrt((2*f1+1) * (2*f0+1)) * get_wigner_6j(j1, ii, f1, f0, 1, j0)


