"""
This module implements the Wavefunction Monte Carlo solution for resolved sideband cooling. To avoid repetition the sim
parameters are loaded from tweezer.py file.
"""
import os.path
import time
from math import sqrt

import numpy as np
import qutip as qt

from tweezer import sb_model, atom, pn0, SIM_TIME, FILEOUT, OUTPUT_DIR

###########################
# USER DEFINED PARAMETERS #
###########################
# number of quantum trajectories
N_TRAJ = 2
# number of data points to save
N_POINTS = 50
# times at which simulation data is recorded.
times = np.linspace(0, SIM_TIME/1000, N_POINTS)

##########################
# END OF USER PARAMETERS #
##########################

# expand to variables
gs_levels = np.asarray(sb_model.gs_levels)
es_levels = np.asarray(sb_model.es_levels)
couplings = sb_model.compute_coupling()
couplings_decay = sb_model.compute_decay_coupling()

sw_start = sb_model.sweep_params['start']
sw_stop = sb_model.sweep_params['stop']
sw_tau = sb_model.sweep_params['time']
laser_s0 = sb_model.sweep_params['s0']

gamma = atom['gamma']

##################################
# QUANTUM COMPONENTS DEFINITIONS #
##################################
# TLS basis
g, e = qt.fock(2, 0), qt.fock(2, 1)

nmax_g = len(gs_levels)
nmax_e = len(es_levels)
nmax = nmax_g + nmax_e
# trap bound states
trap_states = [qt.fock(nmax, i) for i in range(nmax)]

##########################
# HAMILTONIAN DEFINITION #
##########################
# This string defines the time-dependency of the non-constants Hamiltonian terms.
sweep_str = f"(({sw_stop} - {sw_start}) * t / {sw_tau} + {sw_start}) / 2"

# Ground states energies
Hg = sum(
    [gs_levels[n] * qt.tensor(g * g.dag(), ns * ns.dag()) for n, ns in enumerate(trap_states[:nmax_g])]
)
# Non-constant ground state term (cooling laser energy: hbar * omega_c / 2, where hbar=1)
Hgt = [
    sum([qt.tensor(g * g.dag(), ns * ns.dag()) for n, ns in enumerate(trap_states[:nmax_g])]),
    f"+{sweep_str}"
]

# Exited states energies
He = sum(
    [es_levels[n] * qt.tensor(e * e.dag(), ns * ns.dag()) for n, ns in enumerate(trap_states[nmax_g:])]
)
# Non-constant excited state term (cooling laser energy: -hbar * omega_c / 2, where hbar=1)
Het = [
    sum([qt.tensor(e * e.dag(), ns * ns.dag()) for n, ns in enumerate(trap_states[nmax_g:])]),
    f"-{sweep_str}"
]

# Excitations
Vp = sum(
    [sum(
        [couplings[ng, me] * qt.tensor(e * g.dag(), ms * ns.dag()) for ng, ns in enumerate(trap_states[:nmax_g])]
    ) for me, ms in enumerate(trap_states[nmax_g:])]
)
Vp *= sqrt(laser_s0 / 2) * gamma / 2

# De-excitations
Vm = sum(
    [sum(
        [
            couplings[ng, me].conjugate() * qt.tensor(g * e.dag(), ns * ms.dag())
            for ng, ns in enumerate(trap_states[:nmax_g])
        ]
    ) for me, ms in enumerate(trap_states[nmax_g:])]
)
Vm *= sqrt(laser_s0 / 2) * gamma / 2

# Full Hamiltonian
H = [Hg + He + Vp + Vm, Hgt, Het]

# Decay operators
decay_ops = [
    couplings_decay[ng, me] * qt.tensor(g * e.dag(), ns * ms.dag())
    for ng, ns in enumerate(trap_states[:nmax_g])
    for me, ms in enumerate(trap_states[nmax_g:])
]

# Measurement operators (|g, n_g>)
measure_ops = [qt.tensor(g * g.dag(), ns * ns.dag()) for ng, ns in enumerate(trap_states[:nmax_g])]

# Initial state
# This is formally wrong. The initial state should be an incoherent superposition of different states each a weight
# of Pn0[n]. Qutip does not allow for this kind of initial state. The proper way to proceed is to run all trajectory
# simulations with different starting states randomly selected with wights Pn0 (this is what Zoller and Cirac suggest
# in their paper). Unfortunately this slows down the simulation considerably, I believe the problem is mostly on the
# implementation in qutip that requires a lot of header while starting.
psi0 = sum([sqrt(pn0[n]) * qt.tensor(g, trap_states[n]) for n in range(nmax_g)])
# rho0 = sum([pn0[n] * qt.tensor(g, trap_states[n]) * qt.tensor(g, trap_states[n]).dag() for n in range(nmax_g)])

if __name__ == '__main__':
    # solve with Monte Carlo solver
    print(f"Running MC solver with {N_TRAJ} trajectories, saving {N_POINTS} datapoints.")
    options = qt.Options(ntraj=N_TRAJ, store_states=True, average_states=True)

    t_start = time.time()
    sol = qt.mcsolve(
        H,
        psi0,
        tlist=times,
        c_ops=decay_ops,
        e_ops=measure_ops,
        options=options,
        progress_bar=True
    )

    print('Saving ... ', end='')
    qt.qsave(sol, os.path.join(OUTPUT_DIR, FILEOUT + '_WFMC'))
    print('done')
