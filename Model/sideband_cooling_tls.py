import numpy as np
from math import pi

from Model.odt import OpticalTrap

_polarizations = {-1.0, 0.0, 1.0}


class SidebandCoolingModelTLS:
    def __init__(self, trap_model: OpticalTrap, atom, x=0.0, y=0.0):
        self.trap = trap_model
        self.atom = atom

        # cache some stuff
        self.gs_levels = np.real(self.trap.bound_levels('g', x=x, y=y))
        self.es_levels = np.real(self.trap.bound_levels('e', x=x, y=y))

        self.gs_wavefuncs, self.xpartition = self.trap.bound_wavefunctions('g')
        self.es_wavefuncs, _ = self.trap.bound_wavefunctions('e')

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

        self.overlaps = self.compute_overlaps()
        self.expval_kz = self.compute_expval_kz()
        self.couplings_sq = self.compute_coupling_sq()
        self.decay_couplings = self.compute_decay_coupling()

    def _matrix_setup(self, callback):
        """
        Sets up a coupling or decay rate matrix.
        :param callback: function, must take 4 arguments: gs_f, ng, es_f, me (the unravelled ground and exited indices)
        :return:
        """
        gs_len = len(self.gs_levels)
        es_len = len(self.es_levels)

        mat = [[0.0 for _ in range(es_len)] for _ in range(gs_len)]

        for ng in range(gs_len):
            for me in range(es_len):
                mat[ng][me] = callback(ng, me)
        return np.array(mat)

    def scatrate(self, wcool, s0):
        """Computes the scattering rate matrix for given detuning and intensity parameter."""
        def callback(ng, me):
            gamma = self.atom['gamma']

            e_me = self.es_levels[me]
            e_ng = self.gs_levels[ng]

            return s0 * 0.5 * (gamma**2 / 4) / ((e_me - e_ng - wcool)**2 + gamma**2 / 4)
        return self._matrix_setup(callback)

    def compute_overlaps(self):
        """Computes overlap integrals |<n_g|m_e>|^2"""
        dx = self.xpartition[1] - self.xpartition[0]

        def callback(ng, me):
            return np.sum(self.es_wavefuncs[me] * self.gs_wavefuncs[ng]) * dx

        return self._matrix_setup(callback)

    def compute_expval_kz(self):
        """Computes overlap integrals |<n_g|kz|m_e>|^2"""
        dx = self.xpartition[1] - self.xpartition[0]

        def callback(ng, me):
            wlen = self.atom['wlen']
            k = 2 * pi / wlen
            op = 1j * k * self.xpartition
            return np.sum(self.es_wavefuncs[me] * op * self.gs_wavefuncs[ng]) * dx

        return self._matrix_setup(callback)

    def compute_coupling(self):
        """Computes the matrix <m_e|e^{ikx}|n_g> with 1st order expansion of the operator"""
        return self.overlaps + self.expval_kz

    def compute_coupling_sq(self):
        """Computes the matrix |<m_e|e^{ikx}|n_g>|^2 with 1st order expansion of the operator"""
        return np.abs(self.compute_coupling())**2

    def compute_decay_coupling(self):
        """Computes the decay rate of Lindblad jump operators with 1st order expansion of the operator"""
        gamma = self.atom['gamma']
        return gamma/2 * (np.abs(self.overlaps)**2 + 7/16 * np.abs(self.expval_kz)**2)

    def rate_matrix_dyn(self, t: float):
        """Computes the rate equation coefficient matrix by sweeping the laser to time t"""
        wcool, s0 = sweep(self.sweep_params, t)
        return self.rate_matrix(wcool, s0)

    def rate_matrix(self, wcool, s0):
        """Computes the rate equation coefficient matrix for a given laser frequency"""
        # nmax = min(len(self.gs_levels), len(self.es_levels))
        wcool = np.atleast_1d(wcool)
        s0 = np.atleast_1d(s0)

        couplings = self.couplings_sq
        decay_couplings = self.decay_couplings

        gs_len = len(self.gs_levels)

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
