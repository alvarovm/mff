from __future__ import division, print_function

import numpy as np
from scipy.interpolate import InterpolatedUnivariateSpline


class Spline1D(InterpolatedUnivariateSpline):

    def __init__(self, x0, f):
        """

        :param x0: 1 dimensional array
        :param f: 1-dimensional array
        """

        super(Spline1D, self).__init__(x0, f, k=3, ext=3)

    def ev_all(self, x):
        pass


    def ev_forces(self, rs):
        # Force as a function of a configuration
        ds = np.linalg.norm(rs, axis=1, keepdims=True)
        rs_hat = rs / ds

        fs_scalars = - super(Spline1D, self).__call__(ds, nu=1)

        force_vectors = np.sum(fs_scalars * rs_hat, axis=0)

        return force_vectors

    def ev_energy(self, rs):
        # Energy as a function of a configuration
        ds = np.linalg.norm(rs, axis=1, keepdims=True)

        energy_single = super(Spline1D, self).__call__(ds, nu=0)

        tot_energy = np.sum(energy_single, axis=0)

        return tot_energy

    @classmethod
    def from_file(cls, filename):
        data = np.load(filename)
        x_range, energies = data[:, 0], data[:, 1]

        return cls(x_range, energies)

    @classmethod
    def from_matrix(cls, rs, energies):
        return cls(rs, energies)
