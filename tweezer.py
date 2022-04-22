"""
Simulation script for resolved sideband cooling of a TLS in an optical tweezer. Trap, cooling laser, and atom properties
are specified at the beginning of the script. This is the only section the user should modify.
All quantities are specified in SI units.
"""

import os.path
import pickle

import numpy as np
import scipy.constants as sc

from Model.lightbeam import GaussianBeam
from Model.odt import OpticalTweezer
from Model.sideband_cooling_tls import SidebandCoolingModelTLS
from Model.solver import solve, occupation, find_best_n, init_dist

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

SIM_TIME = 3.376e-3  # eta**2 * gamma * t == 5k

times = np.linspace(0, SIM_TIME, N_POINTS)

# Optical tweezer parameters: wavelength (m), waist (m), laser power (W).
trap_options = {
    'wlen': 1064e-9,
    'w0': 0.7e-6,
    'power': 47.5e-3,  # eta_g = 0.2 @1064nm
}

# Cooling laser parameters: wavelength (m), s0 = I / I_sat where I_sat is saturation intensity
# Polarization is used only for multi-level simulations (leave at zero).
cooling_options = {
    'wlen': 671e-9,
    's0': 0.1,
    'polarization': 0,
}

# Properties of the atom/trapped particle.
# Must define: mass, atomic polarizabilities for ground and excited state, a list of ground and excited states,
# wavelength (m) of the main line (used to compute recoil energy), and linewidth gamma (radiants) of the main line
atom = dict(
    m=6 * sc.u,
    polarizabilities={
        'g': 269.574 * alpha_0,
        'e': 191.369 * alpha_0
    },
    states={
        # the 'term' entry can be freely chosen (is just a name).
        # 'energy' refers to the hyperfine energy shift (leave at 0.0 for Two-level systems)
        'ground': [{'term': 'S12', 'energy': 0.0}],
        'exited': [{'term': 'P12', 'energy': 0.0}]
    },
    wlen=671.0e-9,
    gamma=1/27.102e-9
)

# Output filename
FILEOUT = f"sim_tweezer_Li_1064"

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
sb_model = SidebandCoolingModelTLS(optical_trap, atom)
sb_model.sweep_params['s0'] = cooling_options['s0']
sb_model.sweep_params['time'] = SIM_TIME
sb_model.sweep_params['polarization'] = cooling_options['polarization']

# INITIAL DISTRIBUTION
pn0 = init_dist(sb_model)
n0 = occupation(pn0)

print(f'Initial average occupation number: {n0:.3f}')
print()

################################
# BACKUP SIMULATION PARAMETERS #
################################
# to be used for WFMC simulation
params = {
    'couplings': sb_model.compute_coupling(),
    'decay couplings': sb_model.compute_decay_coupling(),
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
    nvt = [occupation(p) for p in pns]
    dist_from_sb = find_best_n(sb_model, nvt, times, atom['gamma'])
    # adjust the cooling laser frequency where the sweep ends.
    sb_model.sweep_params['stop'] -= atom['gamma'] * dist_from_sb

    print(f"Best <n> found with nu = {dist_from_sb:.3f} Gamma from last red sideband\n")

    if dist_from_sb >= 0.1:
        # solve (again) with new cooling laser sweep.
        pns = solve(sb_model, pn0, times)

    # normalize the time axis to eta**2 * gamma (sideband scattering rate)
    time_factor = optical_trap.lamb_dicke('g') ** 2 * atom['gamma']

    #  compute <n>
    nvt = [occupation(p) for p in pns]

    # store to output container
    data = {
        # x-axis
        'times': times,
        'time factor': time_factor,
        # results
        'occupation': nvt,
        'probabilities': pns,
        # stop position
        'stop position': dist_from_sb,
    }

    ###########################
    # SAVE SIMULATION RESULTS #
    ###########################
    with open(os.path.join(OUTPUT_DIR, FILEOUT), 'wb+') as fileout:
        pickle.dump(data, fileout)

    print(f"Data saved to: {FILEOUT}")
