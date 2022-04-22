"""
Module implements the formulae for calculating the polarization of atomic orbitals in an electric field.
These formulae are adapted from Safronova2012 (10.1103/PhysRevA.86.042505).

The values for transition energies and dipole moments are stored in ``./LiE1.csv''. They are stored as a 9-row
matrix, defining the transition starting state and its energy (first 4 columns) the transition end state and its energy
(columns from 4 to 7) and finally the dipole element (8-th column).

The csv is loaded in a DataFrame, then the function get_transitions() repackages the transition info in a dict. Note
that if the transition is upward its energy is positive, else is negative.

The functions alpha0 and alpha2 implement Safronova calculations, they take a list of transition and a wavelength as
inputs. In order to avoid iterating over the LIE1 matrix for every computation of alpha0/alpha2 four specialized
versions of the functions are defined using functools.partial.

The model was validated against data extracted from Safronova's plots in her article.
"""
import numpy as np
import pandas as pd
import scipy.constants as sc
import sympy.physics.wigner as wg
import re

import functools
import os

# physical constants
hbar = sc.hbar
c = sc.c

ANG_MOMENTUM = {
    's': 0,
    'p': 1,
    'd': 2,
    'f': 3
}

# read the transitions matrices
filedir = os.path.dirname(__file__)
li_mat = pd.read_csv(os.path.join(filedir, "LiE1.csv"))


def parse_spectroscopic_term(term):
    # parse second part of state
    fmt = r"^([0-9])(\S)([0-9]\.?[0-9]?)$"

    match = re.match(fmt, term)

    if match is None:
        raise RuntimeError(f'Spectroscopic term ({term}) unrecognized')

    s = float(match.group(1))
    try:
        l = ANG_MOMENTUM[match.group(2)]
    except KeyError:
        l = -1
    j = float(match.group(3))

    return l, j, s


def get_transitions(state, term, trans_mat):
    """
    Selects all transitions (upward and downward) related to state 'state'
    given a transition list (list-like) 'trans_mat'. Each transition is saved
    as a dictionary and appended to a list.
    """
    upward_t = trans_mat[
            (trans_mat["state"] == state) &
            (trans_mat["term"] == term)
            ]

    transitions = []
    for i, t in upward_t.iterrows():
        energy = sc.h * sc.c * (trans_mat.at[i, 'E_p'] - trans_mat.at[i, 'E']) * 10**2
        dip = trans_mat.at[i, 'dip']

        tran = {
            "state": state,
            "term": term,
            "state_p": trans_mat.at[i, 'state_p'],
            "term_p": trans_mat.at[i, 'term_p'],
            "energy": energy,
            "dip": dip
        }
        transitions.append(tran)

    downward_t = trans_mat[
        (trans_mat["state_p"] == state) &
        (trans_mat["term_p"] == term)
        ]

    for i, t in downward_t.iterrows():
        energy = sc.h * sc.c * (trans_mat.at[i, 'E'] - trans_mat.at[i, 'E_p']) * 10**2
        dip = trans_mat.at[i, 'dip']

        tran = {
            "state": state,
            "term": term,
            "state_p": trans_mat.at[i, 'state'],  # mind that this is downward
            "term_p": trans_mat.at[i, 'term'],
            "energy": energy,
            "dip": dip
        }
        transitions.append(tran)

    return transitions


def alpha0(transitions, wlen, use_si=True):
    """
    Generic function for calculating the scalar component of polarizability for
    a set of transitions given as input.
    """
    term = transitions[0]['term']
    _, j, _ = parse_spectroscopic_term(term)

    contribs = np.zeros(len(transitions))

    for i, t in enumerate(transitions):
        de = t['energy']
        dip = t['dip']
        contribs[i] = dip**2 * de / (de**2 - (sc.h * sc.c / wlen)**2)

    pol = np.sum(contribs)
    # with all units in SI except for the dipole element, dimensional
    # analysis yields alpha = a.u./J. Therefore, to get everything in
    # a.u. one has to multiply for the J->a.u. conversion factor
    # which is 4.35...10^-18*)
    pol *= 2.0 / (3 * (2 * j + 1)) * 4.35974417e-18

    if use_si:
        # convert to SI units (this should be related to the value of Bohr radius)
        pol *= sc.h * 2.48832 * 10**(-8)

    return pol


def alpha2(transitions, wlen, use_si=True):
    """
    Generic function for calculating the vectorial component of polarizability
    for a set of transitions given as input.
    """
    term = transitions[0]['term']
    _, j, _ = parse_spectroscopic_term(term)

    c = np.sqrt((5*j * (2*j-1)) / (6*(j+1) * (2*j+1) * (2*j+3)))

    contribs = np.zeros(len(transitions))

    for i, t in enumerate(transitions):
        term = t['term_p']
        _, j1, _ = parse_spectroscopic_term(term)

        de = t['energy']
        dip = t['dip']
        sign = (-1)**(j1 + j + 1)
        contribs[i] = sign * dip**2 * de / (de**2 - (sc.h * sc.c / wlen)**2)
        contribs[i] *= float(wg.wigner_6j(j, 1, j1, 1, j, 2))

    pol = np.sum(contribs)
    # with all units in SI except for the dipole element, dimensional
    # analysis yields [alpha] = a.u./J. Therefore, to get everything in
    # a.u. one has to multiply for the J->a.u. conversion factor
    # which is 4.35...10^-18
    pol *= -4 * c * 4.35974417e-18

    if use_si:
        # convert to SI units (this should be related to the value of Bohr radius)
        pol *= sc.h * 2.48832 * 10**(-8)

    return pol


def alpha(transitions, wlen, use_si=True, vect_sign=1.0):
    """
    Generic function for calculating the polarizability for a set of
    transitions given as input.
    """
    return alpha0(transitions, wlen, use_si) + vect_sign * alpha2(transitions, wlen, use_si)


# level-specialized functions
a_Li2S12 = functools.partial(alpha, get_transitions("2s", "2S0.5", li_mat))
a_Li2P12 = functools.partial(alpha, get_transitions("2p", "2P0.5", li_mat))
a_Li2P32M12 = functools.partial(alpha, get_transitions("2p", "2P1.5", li_mat), vect_sign=-1)
a_Li2P32M32 = functools.partial(alpha, get_transitions("2p", "2P1.5", li_mat), vect_sign=+1)
a_Li3P12 = functools.partial(alpha, get_transitions("3p", "2P0.5", li_mat))
# save them in a dictionary
a_funcs_dict_li = {
    '2S12': a_Li2S12,
    '2P12': a_Li2P12,
    '2P32M12': a_Li2P32M12,
    '2P32M32': a_Li2P32M32,
    '3P12': a_Li3P12
}

if __name__ == '__main__':
    pass