import os.path
import pickle

import numpy as np
import scipy.constants as sc

from Model.wigner_eckart import gamma_hfs
from Model.lightbeam import GaussianBeam
from Model.odt import OpticalTweezer
from Model.sideband_cooling_hfs import SidebandCoolingModelHFS
from Model.solver import solve, occupation, find_best_n, init_dist, get_result_by_state

# physical constants
kb = sc.k
hbar = sc.hbar
pi = np.pi
alpha_0 = sc.h * 2.48832 * 10**(-8)  # conversion from a.u. to SI for atomic polarizability

###########################
# USER DEFINED PARAMETERS #
###########################
do_wfmc = False
# Number of data points to save
N_POINTS = 400

SIM_TIME = 5e-3

times = np.linspace(0, SIM_TIME, N_POINTS)

# Optical tweezer parameters: wavelength (m), waist (m), laser power (W).
trap_options = {
    'wlen': 1064e-9,
    'w0': 0.7e-6,
    'power': 40e-3,
}

# Cooling laser parameters: wavelength (m), s0 = I / I_sat where I_sat is saturation intensity
# 'Polarization' is used to select which transitions will be stimulated.
# 'Shifts' defines the frequency shifts of the cooling laser (for a multi-level system more than 1 frequency may be
# needed for cooling).
cooling_options = {
    'wlen': 323e-9,
    's0': 0.1,
    'polarization': 0,
    'shifts': np.array([152.1, -76.05]) * 1e6 * 2 * pi
}

# Properties of the atom/trapped particle.
# Must define: mass, atomic polarizabilities for ground and excited state, a list of ground and excited states,
# wavelength (m) of the main line (used to compute recoil energy), and linewidth gamma (radiants) of the main line
atom = dict(
    m=6 * sc.u,
    polarizabilities={
        'g': 269.01 * alpha_0,
        'e': 269.01 * alpha_0
    },
    states={
        # the 'term' entry can be freely chosen (is just a name).
        # qt numbers identify the J, F, m_F quantum numbers of the HFS states.
        # 'energy' refers to the hyperfine energy shift of the level (used to compute scattering rates from laser)
        'ground': [
            {'term': 'S12 1', 'quantum numbers': (0.5, 0.5, -0.5), 'energy': -152.1e6 * 2 * pi},
            {'term': 'S12 2', 'quantum numbers': (0.5, 0.5, +0.5), 'energy': -152.1e6 * 2 * pi},
            {'term': 'S12 3', 'quantum numbers': (0.5, 1.5, -1.5), 'energy': +152.1e6 * pi},
            {'term': 'S12 4', 'quantum numbers': (0.5, 1.5, -0.5), 'energy': +152.1e6 * pi},
            {'term': 'S12 5', 'quantum numbers': (0.5, 1.5, +0.5), 'energy': +152.1e6 * pi},
            {'term': 'S12 6', 'quantum numbers': (0.5, 1.5, +1.5), 'energy': +152.1e6 * pi}
        ],
        'exited': [
            {'term': '3P12 1', 'quantum numbers': (0.5, 0.5, -0.5), 'energy': 0.0},
            {'term': '3P12 2', 'quantum numbers': (0.5, 0.5, +0.5), 'energy': 0.0},
            {'term': '3P12 3', 'quantum numbers': (0.5, 1.5, -1.5), 'energy': 0.0},
            {'term': '3P12 4', 'quantum numbers': (0.5, 1.5, -0.5), 'energy': 0.0},
            {'term': '3P12 5', 'quantum numbers': (0.5, 1.5, +0.5), 'energy': 0.0},
            {'term': '3P12 6', 'quantum numbers': (0.5, 1.5, +1.5), 'energy': 0.0},
        ],
    },
    wlen=323.0e-9,
    gamma=7541e3 * 2 * pi
)

# Define Gamma for each atomic transitions (between hyperfine levels)
atom['transitions'] = {}
for es in atom['states']['exited']:
    j1, f1, mf1 = es['quantum numbers']
    for gs in atom['states']['ground']:
        j0, f0, mf0 = gs['quantum numbers']

        for q in [-1.0, 0.0, +1.0]:

            if q not in atom['transitions'].keys():
                atom['transitions'][q] = {}

            if es['term'] not in atom['transitions'][q].keys():
                atom['transitions'][q][es['term']] = {}

            gamma = gamma_hfs(atom['gamma'], j0, f0, mf0, j1, f1, mf1, 1.0, q=q)

            atom['transitions'][q][es['term']][gs['term']] = {
                'gamma': gamma, 'wlen': 671e-9
            }


# Output filename
FILEOUT = f"sim_tweezer_LiHFS_1064"

##########################
# END OF USER PARAMETERS #
##########################

#################
# INITIAL SETUP #
#################
# Create output directory
OUTPUT_DIR = os.path.join(os.path.realpath('.'), 'Output')
if not os.path.exists(OUTPUT_DIR):
    os.mkdir(OUTPUT_DIR)

# Trap beam definition
beam = GaussianBeam(
    wlen=trap_options['wlen'],
    w0=trap_options['w0'],
    i0=2 * trap_options['power'] / pi / trap_options['w0'] ** 2
)

# ODT definition
optical_trap = OpticalTweezer(beam, atom)

# Print trap information
for state in ['g', 'e']:
    print(f"Trap depth ({state}): {optical_trap.trap_depth(state)/kb*1e3:.3f} mK")
    print(f"Trap frequency ({state}): 2xPix{optical_trap.harmonic_pulsation(state)/2/pi*1e-6:.3f} MHz")
    print(f"Lamb-Dicke ({state}): {optical_trap.lamb_dicke(state):.5f}")
    print()

v0g = abs(optical_trap.trap_depth('g'))
v0e = abs(optical_trap.trap_depth('e'))
a = (v0e - v0g) / (v0e + v0g)
print(f"Differential depth: {a * 100:.3f}%")

# Sideband cooling model and sweep parameters (sweep time/power)
sb_model = SidebandCoolingModelHFS(optical_trap, atom)
sb_model.sweep_params['s0'] = np.array([cooling_options['s0'] for _ in cooling_options['shifts']])
sb_model.sweep_params['time'] = SIM_TIME
sb_model.sweep_params['start'] = np.array(
    [sb_model.sweep_params['start'] + shift for shift in cooling_options['shifts']]
)
sb_model.sweep_params['stop'] = np.array(
    [sb_model.sweep_params['stop'] + shift for shift in cooling_options['shifts']]
)
sb_model.sweep_params['polarization'] = cooling_options['polarization']

# INITIAL DISTRIBUTION
pn0 = init_dist(sb_model)
n0 = occupation(pn0)

print(f'Initial average occupation number: {n0:.3f}')
print()

# ravel the initial distribution in a 1D array with length: # of bound states x # of HFS levels.
pn0 = np.tile(pn0, len(atom['states']['ground']))

################################
# BACKUP SIMULATION PARAMETERS #
################################
params = {
    'couplings': sb_model.compute_coupling_matrix(),
    'decay couplings': sb_model.compute_decay_coupling_matrix(),
    'levels': {
        'g': np.asarray(sb_model.gs_levels),
        'e': np.asarray(sb_model.es_levels)
    },
    'wavefunctions': {
        'g': sb_model.gs_wavefuncs,
        'e': sb_model.es_wavefuncs,
        'xpart': sb_model.xpartition,
    },
    'sweep': {
        'start': sb_model.sweep_params['start'],
        'stop': sb_model.sweep_params['stop'],
        'time': sb_model.sweep_params['time'],
        's0': sb_model.sweep_params['s0'],
    },
    # trap characteristics
    'trap': {
        'power': trap_options['power'],
        'waist': trap_options['w0'],
        'wlen': int(trap_options['wlen'] * 1e9),
        'eta': {s: optical_trap.lamb_dicke(s) for s in ['g', 'e']},
        'omega': {s: optical_trap.harmonic_pulsation(s) for s in ['g', 'e']},
        'trap depth': {s: optical_trap.trap_depth(s) for s in ['g', 'e']},
    },
    # atom characteristics
    'atom': atom,
    # initial distribution
    'pn0': pn0
}

# save
with open(os.path.join(OUTPUT_DIR, FILEOUT + "_SIMPARAMS"), 'wb+') as fileout:
    pickle.dump(params, fileout)

print(f"Simulation parameters saved to: {FILEOUT + '_SIMPARAMS'}")

#####################
# ANALYTIC SOLUTION #
#####################
if __name__ == '__main__':
    # run analytical solver
    pns = solve(sb_model, pn0, times)

    # check where n(t) is lowest and re-run the simulation stopping the cooling laser at a better frequency.
    pns_by_state, nvt_by_state = get_result_by_state(pns, atom)
    nvt = nvt_by_state['S12 1']
    dist_from_sb = find_best_n(sb_model, nvt, times, atom['gamma'])
    # adjust the cooling laser frequency where the sweep ends.
    sb_model.sweep_params['stop'] -= atom['gamma'] * dist_from_sb
    print(f"Best <n> found with nu = {dist_from_sb:} Gamma from last red sideband\n")

    if np.any(dist_from_sb >= 0.1):
        # solve (again) with new cooling laser sweep.
        pns = solve(sb_model, pn0, times)

    # normalize the time axis to eta**2 * gamma (sideband scattering rate)
    time_factor = optical_trap.lamb_dicke('g') ** 2 * atom['gamma']

    # organize probabilities by HFS level
    pns_by_state, nvt_by_state = get_result_by_state(pns, atom)

    # store to output container
    data = {
        # x-axis
        'times': times,
        'time factor': time_factor,
        # results
        'occupation': nvt_by_state,
        'probabilities': pns_by_state,
        # stop position
        'stop position': dist_from_sb,
    }

    ###########################
    # SAVE SIMULATION RESULTS #
    ###########################
    with open(os.path.join(OUTPUT_DIR, FILEOUT), 'wb+') as fileout:
        pickle.dump(data, fileout)

    print(f"Data saved to: {FILEOUT}")

