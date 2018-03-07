import numpy as np
import create_MFF_grid
from original import GP_for_MFF
import Kernels
from interpolate_MFF import MBExp
import create_MFF_database


def create_mff(filename, cutoff, nbodies=3, ntr=500, ntest=500, sigma=1.0, grid_start=1.5, grid_spacing=0.02):
    # IMPORT XYZ AND CHOOSE PARAMETERS #
    if nbodies == 3:
        ker = Kernels.ThreeBody(theta=[sigma, cutoff / 10.0])
    elif nbodies == 2:
        ker = Kernels.TwoBody(theta=[sigma, cutoff / 10.0])
    else:
        raise ValueError('Kernel order not understood, use 2 for two-body and 3 for three-body')

    # LAUNCH CONF AND FORCE EXTRACTION #
    print('Extracting database from XYZ file')
    elementslist = create_MFF_database.carve_confs(filename, cutoff)

    # elementslist = [38, None]

    # IMPORT CONFS & FORCES #
    forces = np.load('forces_cut={:.2f}.npy'.format(cutoff))
    confs = np.load('confs_cut={:.2f}.npy'.format(cutoff))

    # CREATE GAUSSIAN PROCESS #
    print('Training the GP module')
    gp = GP_for_MFF.GaussianProcess(kernel=ker, noise=1e-5, optimizer=None)

    # SEPARATE INTO TRAINING AND TESTING CONFIGURATIONS #
    ind = np.arange(len(forces))
    ntot = ntest + ntr
    ind_ntot = np.random.choice(ind, size=ntot, replace=False)
    ind_ntr, ind_ntest = ind_ntot[0:ntr], ind_ntot[ntr:ntot]
    tr_confs, tr_forces = confs[ind_ntr], forces[ind_ntr]
    tst_confs, tst_forces = confs[ind_ntest], forces[ind_ntest]

    # TRAIN AND SAVE GP #
    gp.fit(tr_confs, tr_forces)
    gp.save('GP_{}_ntr_{}_sig_{:.2f}_cut_{:.2f}'.format(nbodies, ntr, sigma, cutoff))

    # TEST GP PERFORMANCE #
    if ntest > 0:
        print('Testing the GP module')
        gp_forces = np.zeros((ntest, 3))
        gp_error = np.zeros((ntest, 3))
        for i in np.arange(ntest):
            gp_forces[i, :] = gp.predict(np.reshape(tst_confs[i], (1, len(tst_confs[i]), 5)))
            gp_error[i, :] = gp_forces[i, :] - tst_forces[i, :]
        print('MAEF on forces: {:.4f} +- {:.4f}'.format(
            np.mean(np.sqrt(np.sum(np.square(gp_error), axis=1))),
            np.std(np.sqrt(np.sum(np.square(gp_error), axis=1)))))

    # BUILD MAPPED GRID FOR M-FF #
    if len(elementslist) == 2:
        mapping = create_MFF_grid.Grids_building(
            d0=grid_start, df=cutoff,
            nr=int(1 + (cutoff - grid_start) / grid_spacing), gp=gp,
            element1=elementslist[0], element2=elementslist[1], nnodes=4)
    elif len(elementslist) == 1:
        mapping = create_MFF_grid.Grids_building(
            d0=grid_start, df=cutoff,
            nr=int(1 + (cutoff - grid_start) / grid_spacing), gp=gp,
            element1=elementslist[0], nnodes=4)
    else:
        raise ValueError('There are more than 2 different elements in the configuration, ' +
                         'remapping is not implemented for these systems yet.')

    all_grids = mapping.build_grids()
    remap_name = ('MFF_{}b_ntr_{}_sig_{:.2f}_cut_{:.2f}.npy'.format(nbodies, ntr, sigma, cutoff))
    np.save(remap_name, all_grids)

    interp = MBExp()
    ff = interp.tri_E_forces_confs
    ef = interp.tri_E_energies_confs
    interp.initialize(remap_name)
    if ntest > 0:
        mff_forces = np.zeros((ntest, 3))
        mff_error = np.zeros((ntest, 3))
        for i in np.arange(ntest):
            mff_forces[i, :] = ff(tst_confs[i])
            mff_error[i, :] = gp_forces[i, :] - mff_forces[i, :]

        print('MAEF on forces: {:.4f} +- {:.4f}'.format(
            np.mean(np.sqrt(np.sum(np.square(gp_error), axis=1))),
            np.std(np.sqrt(np.sum(np.square(gp_error), axis=1)))))


create_mff(filename='C_a_test.xyz', cutoff=3.2, nbodies=3, ntr=2, ntest=10, sigma=1.2, grid_start=1.5, grid_spacing=0.1)
