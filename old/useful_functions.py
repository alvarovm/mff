from __future__ import print_function
import numpy as np
import os.path
import datetime
import random
from GP_custom_gen import GaussianProcess3 as GPvec
# from MLCalculator import MLCalculator as MLC
# from ThreeBodyExpansion_spline import TBExp as TBE

'''
### Choose Directory and files ###
InDir = "Fe_vac"
forces = np.asarray(np.load(os.path.join(InDir,"forces.npy")))
confs  = np.asarray(np.load(os.path.join(InDir,"confs.npy" )))
numconfs = len(forces)


### Set parameters ###
type_of_ker = 'cov_theano'		# Choose Type of Kernel
sigGP = 1.2					    # GP sigma parameter
gam = 2.0					    # GP gamma parameter
ntest = 100						# Number of testing points
ntr = 3						    # Number of training points
rmin, rmax = 2.0, 4.5			# Minimum and maximum distances at which we want to interpolate the force field
nr = 100						# Number of points in the r grid of the remap					
nphi = 2*nr						# Number of points in the phi grid of the remap
'''

### Sample training and testing confs ###
def subsample(confs, forces, ntest, ntr, indices = False):
	ind = np.arange(len(forces))
	ntot = ntest + ntr
	ind_ntot = np.random.choice(ind, size=ntot, replace=False)
	ind_ntr = ind_ntot[0:ntr]
	ind_ntest = ind_ntot[ntr:ntot]
	tr_confs = confs[ind_ntr]
	tr_forces = forces[ind_ntr]
	tst_confs = confs[ind_ntest]
	tst_forces = forces[ind_ntest]
	if indices:
		return (tr_confs, tr_forces, tst_confs, tst_forces, ind_ntr, ind_ntest)
	else:
		return (tr_confs, tr_forces, tst_confs, tst_forces)
	
def mult_subsample(confs1, forces1, ntest1, ntr1, confs2, forces2, ntest2, ntr2, confs3, forces3, ntest3, ntr3, confs4, forces4, ntest4, ntr4, confs5, forces5, ntest5, ntr5, indices = False):
	
	totntr = ntr1+ntr2+ntr3+ntr4+ntr5
	totntest = ntest1+ntest2+ntest3+ntest4+ntest5
	
	# initialize train and test confs
	tr_confs = np.zeros((totntr, len(confs1[0,:,0]), 3))
	tr_forces = np.zeros((totntr, 3))
	tst_confs = np.zeros((totntest, len(confs1[0,:,0]), 3))
	tst_forces = np.zeros((totntest, 3))
    
	# Subsample random training and testing confgurations from shell, core and mixed datasets
	tr_confs[0:ntr1],  tr_forces[0:ntr1],  tst_confs[0:ntest1],  tst_forces[0:ntest1], ind_ntr1, ind_ntest1  = subsample(confs1,  forces1,  ntest1, ntr1, indices = True)
	tr_confs[ntr1:ntr1+ntr2],  tr_forces[ntr1:ntr1+ntr2],  tst_confs[ntest1:ntest1+ntest2],  tst_forces[ntest1:ntest1+ntest2], ind_ntr2, ind_ntest2  = subsample(confs2,  forces2,  ntest2, ntr2, indices = True)
	tr_confs[ntr1+ntr2:ntr1+ntr2+ntr3],  tr_forces[ntr1+ntr2:ntr1+ntr2+ntr3],  tst_confs[ntest1+ntest2:ntest1+ntest2+ntest3],  tst_forces[ntest1+ntest2:ntest1+ntest2+ntest3], ind_ntr3, ind_ntest3  = subsample(confs3,  forces3,  ntest2, ntr3, indices = True)
	tr_confs[ntr1+ntr2+ntr3:totntr-ntr5],  tr_forces[ntr1+ntr2+ntr3:totntr-ntr5],  tst_confs[ntest1+ntest2+ntest3:totntest-ntest5],  tst_forces[ntest1+ntest2+ntest3:totntest-ntest5], ind_ntr4, ind_ntest4  = subsample(confs4,  forces4,  ntest2, ntr4, indices = True)
	tr_confs[totntr-ntr5:totntr],  tr_forces[totntr-ntr5:totntr],  tst_confs[totntest-ntest5:totntest],  tst_forces[totntest-ntest5:totntest], ind_ntr5, ind_ntest5  = subsample(confs5,  forces5,  ntest2, ntr5, indices = True)
	
	if indices:
		ind_tr_tot = []
		ind_tr_tot.append(ind_ntr1)
		ind_tr_tot.append(ind_ntr2)
		ind_tr_tot.append(ind_ntr3)
		ind_tr_tot.append(ind_ntr4)
		ind_tr_tot.append(ind_ntr5)

		ind_test_tot = []
		ind_test_tot.append(ind_ntest1)
		ind_test_tot.append(ind_ntest2)
		ind_test_tot.append(ind_ntest3)
		ind_test_tot.append(ind_ntest4)
		ind_test_tot.append(ind_ntest5)

		return(tr_confs, tr_forces, tst_confs, tst_forces, ind_tr_tot, ind_test_tot)
	else:
		return(tr_confs, tr_forces, tst_confs, tst_forces)
	
	
	
### Define and train a Gaussian Process ###
def train_GP(tr_confs, tr_forces, type_of_ker, sigGP, gam):
	gp = GPvec( ker=[ 'id'], fvecs =[type_of_ker] ,nugget = 1e-5, theta0=np.array([1.0]), sig =sigGP, gamma = gam, bounds = ((0.1,10.),), optimizer= None , calc_error = False, eval_grad = False)
	t0_train = datetime.datetime.now()
	gp.fit(tr_confs, tr_forces)
	tf_train = datetime.datetime.now()
	print("Training computational time is", (tf_train-t0_train).total_seconds())
	return gp
	
	
### Test the performance of the Gaussian Process ###
def test_GP_forces(gp, tst_confs, tst_forces):
	ntest = len(tst_forces)
	gp_error = np.zeros((ntest, 3))
	gp_forces = np.zeros((ntest, 3))
	t0_predict = datetime.datetime.now()
	for j in np.arange(ntest):
		gp_forces[j] = gp.predict(np.reshape((tst_confs[j]), (1, len(tst_confs[j]), 3))) 
		gp_error[j] = gp_forces[j] - tst_forces[j]
	tf_predict = datetime.datetime.now()
	print("GP Prediction computational time is", (tf_predict-t0_predict).total_seconds())
	return(gp_forces, gp_error)
	
	
### Create a remapping of the Gaussian Process on a grid of the chosen density and range ###
def train_remap(gp, rmin, rmax, nr, nphi):
	rgrid = np.linspace(rmin, rmax, nr)
	phigrid = np.linspace(0.0, np.pi, nphi)
	t0_remap = datetime.datetime.now()
	exp = TBE(gp)
	exp.TB_S_fit(rs = rgrid, ts = phigrid, in_force = [] ,imp = False, plot = False)
	tf_remap = datetime.datetime.now()
	print("Remap computational time is", (tf_remap-t0_remap).total_seconds())
	ff = exp.tri_force_conf
	ef = exp.energy
	return (ff,ef)
	
### Train remap process and save it with an appropriate name
def train_and_save_remap(gp, rmin, rmax, nr, nphi, folder, ntr):
	rgrid = np.linspace(rmin, rmax, nr)
	phigrid = np.linspace(0.0, np.pi, nphi)
	t0_remap = datetime.datetime.now()
	exp = TBE(gp)
	exp.TB_S_fit(rs = rgrid, ts = phigrid, in_force = [] ,imp = False, plot = False)
	tf_remap = datetime.datetime.now()
	print("Remap computational time is", (tf_remap-t0_remap).total_seconds())
	name = ('RemappedGP_%s_ntr=_%i_nr=_%i_nphi=_%i_rmin=_%f_rmax=_%f.npy' % (folder, ntr, nr, nphi, rmin, rmax))
	exp.save_force_field(name)
	return (name)	
	
### Test the remapping against the true forces and the GP ###
def test_remap_forces(ff, tst_confs, tst_forces, gp_forces = None):
	ntest = len(tst_forces)
	remap_forces = np.zeros((ntest,3))
	error_ontrue = np.zeros((ntest,3))
	error_onGP =np.zeros((ntest,3))
	t0_predict = datetime.datetime.now()
	for j in np.arange(ntest):
		remap_forces[j] = ff(tst_confs[j])
		error_ontrue[j] = remap_forces[j] - tst_forces[j]
		error_onGP[j] = remap_forces[j] - gp_forces[j]
	tf_predict = datetime.datetime.now()
	print("Remap Prediction computational time is", (tf_predict-t0_predict).total_seconds())
	return (remap_forces, error_ontrue, error_onGP)
	

### Relax the structure within the threshold
def optimize(atoms, maxsteps, savefile, threshold = 0.005):
	n = len(atoms)
	positions = atoms.get_positions()
	for i in np.arange(maxsteps):
		print('step', i)
		forces = atoms.get_forces()
		np.savetxt(savefile, np.concatenate((atoms.get_positions(), forces), axis = 1), fmt='%.8f', delimiter='       ', newline='\n', header=('%i \nLattice="0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0" Properties=pos:R:3:force:R:3 pbc="F F F"' %(n)), comments = '')
		if (max(np.sqrt(np.einsum('nd->n', np.square(forces)))) <= threshold):
			savefile.close()
			return (positions, forces, i)
		else:
			atoms.set_positions(positions + 0.05*forces)
			positions = atoms.get_positions()
	savefile.close()
	return(0, 0, i)

def absol(force_error):
	return np.sqrt(np.einsum('ij->i', np.square(force_error)))
