import numpy as np
from math import pi, sqrt

from Model.odt import OpticalTrap

_polarizations = {-1.0, 0.0, 1.0}


class SidebandCoolingModelHFS:
    def __init__(self, trap_model: OpticalTrap, atom, x=0.0, y=0.0):
        self.trap = trap_model
        self.atom = atom

        # cache some stuff
        self.gs_levels = np.real(self.trap.bound_levels('g', x=x, y=y))
        self.es_levels = np.real(self.trap.bound_levels('e', x=x, y=y))

        self.gs_wavefuncs, self.xpartition = self.trap.bound_wavefunctions('g')
        self.es_wavefuncs, _ = self.trap.bound_wavefunctions('e')

        # store atomic states and assign a number 0 to N
        self._atomic_ground_states = [s for s in atom['states']['ground']]
        self._atomic_exited_states = [s for s in atom['states']['exited']]

        # compute sweep constants
        # compute laser start and stop frequencies
        nmax = min(len(self.gs_levels), len(self.es_levels))
        self.red_sidebands = np.array(self.es_levels[:nmax-1]) - np.array(self.gs_levels[1:nmax])

        self.sweep_params = {
            'start': self.red_sidebands[-1],
            'stop':  self.red_sidebands[0],
            'time': 5e-3,
            's0': 0.1,
            'polarization': 0
        }

        self.couplings = np.abs(self.compute_coupling_matrix())**2
        self.decay_couplings = np.abs(self.compute_decay_coupling_matrix())**2

    def _matrix_setup(self, callback):
        """
        Sets up a coupling or decay rate matrix.
        :param callback: function, must take 4 arguments: gs_f, ng, es_f, me (the unravelled ground and exited indices)
        :return:
        """
        gs_shape = (len(self._atomic_ground_states), len(self.gs_levels))
        es_shape = (len(self._atomic_exited_states), len(self.es_levels))
        gs_len = gs_shape[0] * gs_shape[1]
        es_len = es_shape[0] * es_shape[1]

        mat = [[0.0 for _ in range(es_len)] for _ in range(gs_len)]

        for gs_index in range(gs_len):
            for es_index in range(es_len):
                gs_f, ng = np.unravel_index(gs_index, gs_shape)
                es_f, me = np.unravel_index(es_index, es_shape)

                mat[gs_index][es_index] = callback(gs_f, ng, es_f, me)
        return np.array(mat)

    def scatrate(self, wcool, s0):
        """Computes the scattering rate matrix for given detuning and intensity parameter."""
        def callback(gs_f, ng, es_f, me):
            gs_term = self._atomic_ground_states[gs_f]['term']
            es_term = self._atomic_exited_states[es_f]['term']

            q = self.sweep_params['polarization']

            try:
                gamma = self.atom['transitions'][q][es_term][gs_term]['gamma']
            except KeyError:
                gamma = 0.0

            if not gamma:
                return 0.0

            e_me = self._atomic_exited_states[es_f]['energy'] + self.es_levels[me]
            e_ng = self._atomic_ground_states[gs_f]['energy'] + self.gs_levels[ng]

            return s0 * 0.5 * (gamma**2 / 4) / ((e_me - e_ng - wcool)**2 + gamma**2 / 4)
        return self._matrix_setup(callback)

    def compute_coupling_matrix(self):
        """Computes the matrix |<m_e|e^{ikx}|n_g>|^2 with 1st order expansion of the operator"""
        dx = self.xpartition[1] - self.xpartition[0]

        def callback(gs_f, ng, es_f, me):
            gs_term = self._atomic_ground_states[gs_f]['term']
            es_term = self._atomic_exited_states[es_f]['term']

            q = self.sweep_params['polarization']

            try:
                # todo: this is actually wrong: it should be the cooling light wlen (not a problem for TLS)
                wlen = self.atom['transitions'][q][es_term][gs_term]['wlen']
            except KeyError:
                raise
            k = 2 * pi / wlen
            op = 1 + 1j * k * self.xpartition
            return np.sum(self.es_wavefuncs[me] * op * self.gs_wavefuncs[ng]) * dx

        return self._matrix_setup(callback)

    def compute_decay_coupling_matrix(self):
        """Computes the decay rate of Lindblad jump operators with 1st order expansion of the operator"""
        dx = self.xpartition[1] - self.xpartition[0]

        def callback(gs_f, ng, es_f, me):
            gs_term = self._atomic_ground_states[gs_f]['term']
            es_term = self._atomic_exited_states[es_f]['term']

            _, _, mf0 = self._atomic_ground_states[gs_f]['quantum numbers']
            _, _, mf1 = self._atomic_exited_states[es_f]['quantum numbers']

            q = mf1 - mf0

            if q not in _polarizations:
                return 0.0

            try:
                gamma = self.atom['transitions'][q][es_term][gs_term]['gamma']
                # todo: this is actually wrong: it should be the cooling light wlen (not a problem for TLS)
                wlen = self.atom['transitions'][q][es_term][gs_term]['wlen']
            except KeyError:
                raise
            k = 2 * pi / wlen
            op = 1 + 1j * k * self.xpartition * sqrt(2/5)
            return sqrt(gamma / 2) * np.sum(self.gs_wavefuncs[ng] * op * self.es_wavefuncs[me]) * dx

        return self._matrix_setup(callback)

    def rate_matrix_dyn(self, t: float):
        """Computes the rate equation coefficient matrix by sweeping the laser to time t"""
        wcool, s0 = sweep(self.sweep_params, t)
        return self.rate_matrix(wcool, s0)

    def rate_matrix(self, wcool, s0):
        """Computes the rate equation coefficient matrix for a given laser frequency"""
        # nmax = min(len(self.gs_levels), len(self.es_levels))
        wcool = np.atleast_1d(wcool)
        s0 = np.atleast_1d(s0)

        couplings = self.couplings
        decay_couplings = self.decay_couplings

        gs_shape = (len(self._atomic_ground_states), len(self.gs_levels))
        gs_len = gs_shape[0] * gs_shape[1]

        mat = [[0.0]*gs_len for _ in range(gs_len)]

        for laser_freq, laser_s0 in zip(wcool, s0):
            scatrates = self.scatrate(laser_freq, laser_s0)
            for ng in range(gs_len):
                for mg in range(gs_len):

                    if ng == mg:
                        continue

                    rate = np.sum(decay_couplings[ng] * couplings[mg] * scatrates[mg])

                    mat[ng][mg] += rate
        mat = np.array(mat, dtype=np.float64)

        mat -= np.diag(np.sum(mat, axis=0))
        # todo: this should be soft-coded
        mat[np.abs(mat) <= 1e-10] = 0.0

        return mat


def sweep(sweep_params, t):
    if t > sweep_params['time']:
        return sweep_params['stop'], 0.0
    ff = (sweep_params['stop'] - sweep_params['start']) * t / sweep_params['time']
    ff += sweep_params['start']
    return ff, sweep_params['s0']



