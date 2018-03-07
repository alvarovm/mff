import numpy as np
from scipy import interpolate
from itertools import combinations
from mlfmapping import Spline1D, Spline3D

# from numba import jit

D = 3


class MBExp:

    def __init__(self):
        None

    @staticmethod
    def vectorize(rs):
        n, m = rs.shape
        dtype = rs.dtype

        # number of outputs
        n_out = n * (n - 1) / 2

        # Allocate arrays
        r1 = np.zeros([n_out, 1], dtype=dtype)
        r2 = np.zeros([n_out, 1], dtype=dtype)

        r1_hat = np.zeros([n_out, m], dtype=dtype)
        r2_hat = np.zeros([n_out, m], dtype=dtype)

        cosphi = np.zeros([n_out, 1], dtype=dtype)

        # Calculate the norm and the normal vectors
        # ds = np.linalg.norm(rs, axis=1, keepdims=True)
        # rs_hat = rs / ds
        ds = np.sqrt(np.einsum('nd, nd -> n', rs, rs))
        rs_hat = np.einsum('nd, n -> nd', rs, 1. / ds)

        for i, ((r1_i, r1_hat_i), (r2_i, r2_hat_i)) in enumerate(combinations(zip(ds, rs_hat), r=2)):
            r1[i], r2[i] = r1_i, r2_i
            r1_hat[i, :], r2_hat[i, :] = r1_hat_i, r2_hat_i
            cosphi[i] = np.clip(np.dot(r1_hat_i, r2_hat_i), -1.0, 1.0)

        return r1, r2, r1_hat, r2_hat, cosphi

    # ~ def test_interpolation(self):
    # ~ r1 = np.array([
    # ~ 3.9195691981972485,
    # ~ 2.4890138181742203,
    # ~ 2.4890138181742203])

    # ~ r2 = np.array([
    # ~ 2.3306133539159548,
    # ~ 2.3306133539159548,
    # ~ 3.9195691981972485])

    # ~ phi = np.array([
    # ~ 1.7882151827330697,
    # ~ 2.1160693001291131,
    # ~ 0.67010430564407197])

    # ~ e, de_dr1, de_dr2, de_dphi = self.interpx.ev(r1, r2, phi)
    # ~ for e, de_dr1, de_dr2, de_dphi in zip(e, de_dr1, de_dr2, de_dphi):
    # ~ print(e, de_dr1, de_dr2, de_dphi)

    # ~ for r1_i, r2_i, phi_i in zip(r1, r2, phi):
    # ~ e, de_dr1, de_dr2, de_dphi = self.interpx.ev(r1_i, r2_i, phi_i)
    # ~ print(e, de_dr1, de_dr2, de_dphi)

    def initialize(self, remap2_name, remap3_name):
        self.pair_interp = Spline1D.from_file(remap2_name)
        self.tri_interp = Spline3D.from_file(remap3_name)

    def PW_S_fit_fromEgrid(self, rs_energies, k=3):  # Pairwise, spline fit of order k TO BE DEBUGGED

        xgrid, energies = rs_energies[:, 0], rs_energies[:, 1]
        interp = interpolate.InterpolatedUnivariateSpline(xgrid, energies, k=k, ext=3)
        self.interp = interp

    ### grid calculations only ###

    def PW_E_get_gridpoints(self, gp, rs):

        npoints = len(rs)
        confs = np.zeros((npoints, 1, D))
        confs[:, 0, 0] = rs
        energies = gp.predict_energy(confs)

        return np.vstack((rs, energies)).T

    def TB_E_get_gridpoints(self, gp, rs, ts):

        # fitting a pairwise first (for faster evaluation later and for decomposition)
        rs_energies = self.PW_E_get_gridpoints(gp, rs)
        print('Shape of energy grid', rs_energies.shape)

        self.PW_S_fit_fromEgrid(rs_energies, k=5)

        #

        # creating the configurations
        nrpoints = len(rs)
        rgrid = rs
        print('Number of r grid points', nrpoints)
        nphipoints = len(ts)
        print('Number of phi grid points', nphipoints)
        phigrid = ts

        confs = np.zeros((nrpoints * nrpoints * nphipoints, 2, D))
        print('Number of grid points', len(confs))
        confs[:, 0, 0] = np.repeat(rgrid, (nrpoints * nphipoints))
        confs[:, 1, 0] = np.tile(np.reshape(np.outer(rgrid, np.cos(phigrid)), nphipoints * nrpoints), nrpoints)
        confs[:, 1, 1] = np.tile(np.reshape(np.outer(rgrid, np.sin(phigrid)), nrpoints * nphipoints), nrpoints)

        # pairwise contributions

        pair_energies = self.pair_energies_confs(confs)

        #

        from pathos.multiprocessing import ProcessingPool
        n = len(confs)

        if __name__ == 'ManyBodyExpansion_2':
            nods = 16
            print('Using %i cores for the remapping' % (nods))
            pool = ProcessingPool(nodes=nods)
            splitind = np.zeros(nods + 1)
            factor = (n + (nods - 1)) / nods
            splitind[1:-1] = [(i + 1) * factor for i in np.arange(nods - 1)]
            splitind[-1] = n
            splitind = splitind.astype(int)
            clist = [confs[splitind[i]:splitind[i + 1]] for i in np.arange(nods)]
            result = np.array(pool.map(gp.predict_energy, clist))
            result = np.concatenate(result)

        all_energies = result
        # all_energies = np.reshape(all_energies, (all_energies.shape[0]*all_energies.shape[1])) # CHANGE
        three_energies = all_energies - pair_energies

        three_energies = np.reshape(three_energies, (nrpoints, nrpoints, nphipoints))

        grid_energies = np.zeros(((nrpoints, nrpoints, nphipoints, 3)))
        grid_energies[:, :, :, 0], grid_energies[:, :, :, 1] = rs[:, None, None], ts[None, None, :]
        grid_energies[:, :, :, 2] = three_energies

        return rs_energies, grid_energies

    ### prediction ###

    def pair_potential_scalars(self, d0, df, Delta_d=0.1):  # Potential Visualization
        interp = self.interp
        npoints = round((df - d0) / Delta_d)
        distances = np.linspace(df, d0, npoints)

        forces = interp(distances)
        energies = np.cumsum(forces * Delta_d)
        return - np.flipud(energies)

    def pair_energy_conf(self, rs):  # Force as a function of a configuration
        interp = self.interp
        ds = np.sqrt(np.einsum('nd, nd -> n', rs, rs))
        ens = interp(ds)
        en = np.sum(ens)

        return en

    def pair_energies_confs(self, confs):  # Forces as a function of configurations
        interp = self.interp

        pair_energy_conf = self.pair_energy_conf
        energies = np.zeros(len(confs))
        for c in np.arange(len(confs)):
            rs = np.array((confs[c]))
            energy = pair_energy_conf(rs)
            energies[c] = energy

        return energies

    def pair_E_forces_confs(self, confs):  # Forces as a function of configurations
        forces = np.zeros((len(confs), D))
        for c in np.arange(len(confs)):
            rs = np.array((confs[c]))
            force = self.pair_interp.evolute(rs)(rs)
            forces[c] = force

        return forces

    ###############################
    ### THESE ARE THE GOOD ONES ###
    ###############################

    def tri_E_forces_confs(self, confs):  # Forces as a function of configurations
        forces = np.zeros((len(confs), D))
        for c in np.arange(len(confs)):
            rs = np.array((confs[c]))
            force_3 = self.tri_interp.ev_forces(rs)
            force_2 = self.pair_interp.ev_forces(rs)
            forces[c] = force_3 + force_2

        return forces

    def tri_E_energies_confs(self, confs):  # Energies as a function of configurations
        energies = np.zeros(len(confs))
        for c in np.arange(len(confs)):
            rs = np.array((confs[c]))
            energy_3 = self.tri_interp.ev_energy(rs)
            energy_2 = self.pair_interp.ev_energy(rs)
            energies[c] = energy_2 / 2.0 + energy_3 / 3.0

        return energies
