import numpy as np
from ManyBodyExpansion_2 import MBExp
import useful_functions as us
import os.path
import datetime
import matplotlib.pyplot as plt
from GP_custom_gen import GaussianProcess3 as GPvec


def do_the_remap(gp, rs, ts, remap2, remap3, index):
    rmin, rmax = rs[0], rs[-1]  # Minimum and maximum distances at which we want to interpolate the force field
    nr = len(rs)  # Number of points in the r grid of the remap
    nt = len(ts)  # Number of points in the phi grid of the remap						# Number of training points

    # Remap and save #
    exp = MBExp()
    t0_remap = datetime.datetime.now()
    rs_energies, grid_energies = exp.TB_E_get_gridpoints(gp, rs, ts)
    if index == 0:
        np.save(remap2, rs_energies)
    else:
        remap2 = []
    np.save(remap3, grid_energies)
    tf_remap = datetime.datetime.now()
    print("Remap computational time is", (tf_remap - t0_remap).total_seconds())
    print('start and end of the theta grid', ts[0], ts[-1])
    return (remap2, remap3)


### Set parameters ###
type_of_ker = '3b'  # Choose Type of Kernel
sigGP = 1.2  # GP sigma parameter
gamGP = 2.0  # GP Gamma parameter
ntest = 5  # Number of testing points
ntr = 3
r0 = 1.5
rf = 10.0
nr = 10
nt = 4 * nr
splits = 1
database = 'test'
gpname = ('GP_%s_struct_%s_ntr=_%i_sig=%f_gam=%f.npy' % (type_of_ker, database, ntr, sigGP, gamGP))

### Importing data from simulation ###
InDir = "Database"
forces = np.asarray(np.load(os.path.join(InDir, "forces.npy"), encoding='latin1'))
confs = np.asarray(np.load(os.path.join(InDir, "confs.npy"), encoding='latin1'))
for i in np.arange(len(confs)):
    confs[i] = confs[i][:, :3]

gp = GPvec(ker=['id'], fvecs=[type_of_ker], m_theta0=[40.0], nugget=1e-5, theta0=np.array([1.]), sig=sigGP, gamma=gamGP,
           bounds=((0.1, 10.),), optimizer=None, calc_error=False, eval_grad=False)
tr_confs, tr_forces, tst_confs, tst_forces = us.subsample(confs, forces, ntest, ntr)

# Train GP #
if os.path.isfile(os.path.join(InDir, gpname)):
    gp.load(os.path.join(InDir, gpname))

else:
    gp = us.train_GP(tr_confs, tr_forces, type_of_ker, sigGP, gamGP)
    gp.save(os.path.join(InDir, gpname))

# SPLIT REMAPS #
rs = np.linspace(r0, rf, nr)
ts = np.linspace(0.0, np.pi, nt)
remap_names = []
print('Total number of grid points', nr * nr * nt)
for i in np.arange(splits):
    remap2_name = ('Remap2_struct_%s_ntr=_%i_nr=%i_nt=%i_r=%f-%f_part%i.npy' % (database, ntr, nr, nt, r0, rf, i))
    remap3_name = ('Remap3_struct_%s_ntr=_%i_nr=%i_nt=%i_r=%f-%f_part%i.npy' % (database, ntr, nr, nt, r0, rf, i))
    print('The split count is at', i)
    remap_names.append(do_the_remap(gp, rs, ts[i * nt // splits:(i + 1) * nt // splits], remap2_name, remap3_name, i))

# GLUE REMAPS TOGETHER # 
remap_2 = np.zeros(nr)
remap_3 = np.zeros((nr, nr, nt, 3))
for i in np.arange(splits):
    if (i == 0):
        remap_2 = np.load(remap_names[0][0])
    remap_3[:, :, i * nt // splits:(i + 1) * nt // splits, :] = np.load(remap_names[i][1])

# SAVE REMAP # 
remap2_name = ('Remap2_struct_%s_ntr=_%i_nr=%i_nt=%i_r=%f-%f.npy' % (database, ntr, nr, nt, r0, rf))
remap3_name = ('Remap3_struct_%s_ntr=_%i_nr=%i_nt=%i_r=%f-%f.npy' % (database, ntr, nr, nt, r0, rf))
np.save(remap2_name, remap_2)
np.save(remap3_name, remap_3)

# DELETE REMAP PIECES # 
for i in np.arange(splits):
    if i == 0:
        remap2_piece = ('Remap2_struct_%s_ntr=_%i_nr=%i_nt=%i_r=%f-%f_part%i.npy' % (database, ntr, nr, nt, r0, rf, i))
        os.remove(remap2_piece)
    remap3_piece = ('Remap3_struct_%s_ntr=_%i_nr=%i_nt=%i_r=%f-%f_part%i.npy' % (database, ntr, nr, nt, r0, rf, i))
    os.remove(remap3_piece)

remap2_name = ('Remap2_struct_%s_ntr=_%i_nr=%i_nt=%i_r=%f-%f.npy' % (database, ntr, nr, nt, r0, rf))
remap3_name = ('Remap3_struct_%s_ntr=_%i_nr=%i_nt=%i_r=%f-%f.npy' % (database, ntr, nr, nt, r0, rf))

# Interpolate # 
rs_energies = np.load(remap2_name)
grid_energies = np.load(remap3_name)
exp = MBExp()
exp.initialize(remap2_name, remap3_name)

# Predict # 
ff = exp.tri_E_forces_confs
mapped = ff(tst_confs)
preds = gp.predict(tst_confs)

# Confront errors #
errors_remap_GP = np.sqrt(np.sum((preds - mapped) ** 2, axis=1))
errors_remap_true = np.sqrt(np.sum((tst_forces - mapped) ** 2, axis=1))
errors_GP_true = np.sqrt(np.sum((preds - tst_forces) ** 2, axis=1))

print('Average remapping error', np.mean(errors_remap_GP))
print('Average GP error', np.mean(errors_GP_true))
print('Average remap error on true data', np.mean(errors_remap_true))
if False:
    plt.plot(preds, mapped, 'ro')
    plt.plot(preds, preds, 'b-')
    plt.xlabel("GP 3B force")
    plt.ylabel("Derivative 3B force")
    plt.show()
