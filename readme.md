## Resolved Sideband Cooling Simulation

Code for simulation of resolved sideband cooling of two-level or multi-level
particles trapped using optical dipole traps.

Currently, only optical tweezers have been implemented.

User should modify the simulation parameters at the beginning of tweezer.py
(for two-level systems) or tweezer_hfs.py (for multi-level particles) and execute
such files. Corresponding jupyter notebooks are provided to ease data
visualization.

An alternative wavefunction Monte Carlo simulation is provided in tweezer_wfmc.py
(only for two-level systems). This simulation uses the same parameters written in
tweezer.py.

The chapter of my PhD thesis where the mathematics and physics of the model are
explained is provided under the Docs folder.

