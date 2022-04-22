"""
Modules implement functions for solving the sideband cooling model using analytical approximation.
"""

import numpy as np
import scipy.constants as sc
import qutip as qt
from tqdm import tqdm

from Model.sideband_cooling_hfs import sweep


###################
# UTILS FUNCTIONS #
###################
def occupation(p):
    """Computes the average occupation number for a given list of occupation probabilities"""
    return np.real(np.dot(range(len(p)), p) / sum(p))


def boltzman_dist(en, temp):
    p = np.exp(-en / sc.k / temp)
    Q = np.sum(np.exp(-en / sc.k / temp))

    return np.real(p / Q)


def init_dist(model):
    nmax = len(model.gs_levels)
    rho0 = qt.thermal_dm(nmax, nmax)
    return rho0.diag()


###########################
# ANALYTIC SOLVER FOR TLS #
###########################
def solve_diff_eq(mat, p0, dt):
    """
    Solves the differential equation dP/dt = M * P, where P is a column vector containing the occupation probabilities,
    M is a matrix relating P to its time derivative. This function should be called multiple times in a recursive manner
    to propagate the solution over multiple time steps.
    :param mat: np.array, matrix M.
    :param p0: np.array, initial occupation probabilities at t=0 (before cooling).
    :param dt: float, time step.
    :return: np.array, integrated occupation probabilities (over 1 time step).
    """
    eigvals, eigvects = np.linalg.eig(mat)
    c = np.linalg.inv(eigvects) @ p0
    return eigvects @ (c * np.exp(eigvals * dt))


def solve(model, p0, times):
    """
    Solves the Sideband Cooling model by integrating the associated differential equation over a series of time steps.
    :param model: SidebandCooling class, model to be solved.
    :param p0: np.array, initial occupation probabilities at t=0 (before cooling).
    :param times: np.array, partition of the time axis, which is used to integrate step-wise the model's diff. equation.
    :return: np.array, integrated occupation probabilities.
    """
    pns = [p0]
    with tqdm(total=len(times), desc='Running analytic solver', ) as pbar:
        for i, t in enumerate(times[1:], start=1):
            dt = t - times[i - 1]

            matrix = model.rate_matrix_dyn(t)
            pn = np.real(solve_diff_eq(matrix, pns[-1], dt))

            pns.append(pn / sum(pn))

            pbar.update()

    return np.real(pns)


def find_best_n(model, ns, times, gamma):
    """Determines where <n> is at it lowest and returns the corresponding cooling laser frequency."""
    i_best = int(np.argmin(ns))
    # find where the laser was
    t_min = times[i_best]
    laser, _ = sweep(model.sweep_params, t_min)
    # compute distance from last sb
    dist_from_sb = (model.sweep_params['stop'] - laser) / gamma
    return dist_from_sb


###########################
# ANALYTIC SOLVER FOR HFS #
###########################
def get_population(p, gs_index, gs_num):
    """
    Extracts the integrated occupation probabilities of a specific HFS level from a 1D list of all occupation
    probabilities of all HFS levels.
    :param p: np.array containing occupation probabilities for all HFS levels.
    :param gs_index: int identifying which HFS level to extract
    :param gs_num: int, # of HFS levels
    :return: occupation probabilities associated to the given HFS level.
    """
    index_shape = (gs_num, len(p) // gs_num)
    i0 = np.ravel_multi_index((gs_index, 0), index_shape)
    i1 = np.ravel_multi_index((gs_index, index_shape[1] - 1), index_shape)
    pn = p[i0:i1+1]
    return pn / np.sum(pn)


def get_result_by_state(pns, atom):
    """
    Unravels the occupation probabilities array separating the occupation probabilities associated to different HFS
    levels.
    :param pns: np.array, raveled list of occupation probabilities
    :param atom: dict defining the atom ground HFS states
    :return: 2-tuple containing dictionaries. Map HFS ID to occupation probabilities or average occupation number.
    """
    gs_num = len(atom['states']['ground'])
    pns_by_state = {
        state['term']: [
            get_population(pn, i, gs_num) for pn in pns
        ] for i, state in enumerate(atom['states']['ground'])
    }

    nvt_by_state = {
        state: [occupation(p) for p in pn] for state, pn in pns_by_state.items()
    }
    return pns_by_state, nvt_by_state
