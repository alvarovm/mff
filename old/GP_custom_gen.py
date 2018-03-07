"""
Gaussian process regression module. Inspired by sklearn.gaussian_process module and stripped out naked to the bare minimum
"""

from __future__ import print_function, division
from scipy import linalg as LA
from scipy.linalg import cholesky, cho_solve
import scipy as sp
import numpy as np
import scipy.spatial.distance as spdist
from scipy.optimize import fmin_l_bfgs_b
from scipy.optimize import minimize
from scipy.spatial.distance import cdist
from scipy.special import iv
from sklearn.preprocessing import normalize
import theano
from six.moves import cPickle
# from transforms3d import axangles as tr
import os.path

theano.config.cxx = ("/usr/bin/g++")

f = open('Hessian_3b.save', 'rb')
Hessian3c = cPickle.load(f, encoding='latin-1')
f.close()

'''
f = open('Hessian_exp.save', 'rb')
Hessian3e = cPickle.load(f)
f.close()
'''
f = open('Grad_3b.save', 'rb')
Grad3c = cPickle.load(f, encoding='latin-1')
f.close()
'''
f = open('Grad_cut_exp.save', 'rb')
Grad3c_exp = cPickle.load(f)
f.close()

f = open('Hessian_cut_5body.save', 'rb')
Hessian5c = cPickle.load(f)
f.close()

f = open('Hessian_cut_5body.save', 'rb')
Hessian5c = cPickle.load(f)
f.close()

f = open('Hessian_3b_nU.save', 'rb')
Hessian3nu = cPickle.load(f)
f.close()

f = open('Hessian_2b.save', 'rb')
Hessian2c = cPickle.load(f)
f.close()
'''

"""
# for EAM prior
from ase import Atoms
from ase.calculators.eam import EAM
mishin = EAM(potential='DFT/Fe_Ni_EAM/Fe_2.eam.alloy' ,  lattice = ['bcc'], a = [2.83*2*2])
mishin.write_potential('new.eam.alloy')
calc = mishin
"""

MACHINE_EPSILON = sp.finfo(sp.double).eps

D = 3  # number of dimensions of inpud data


##### useful functions #####


def f_cut(r, r_c):  # standard cutoff function
    result = 0.5 * (np.cos(np.pi * r / r_c) + 1.)
    return result


def f_cut2(r, r_c):  # arctan cutoff function
    result = 0.75 - np.arctan(1000 * (r - 0.99 * r_c)) / np.pi
    return result


def f_cut3(r, r_c, sig=0.2):  # Bump cutoff cunction
    result = np.exp(sig / (r - r_c)) * heaviside(r_c - r)
    return result


def heaviside(x):  # Standard Heaviside function
    return (0.5 * (np.sign(x) + 1))


def tripl_cut(r1, r2, r12, rc=2.5):  # Inputs must be all n*n, function is 1 if ij is a valid triplet
    func = heaviside(1.5 - heaviside(r1 - rc) - heaviside(r2 - rc) - heaviside(r12 - rc))
    # Use this if you want to include "triplets" that are nothing but pairs ALSO within the second cutoff radius
    func = heaviside(func + (1 - np.sign(r12) - 0.5))  #
    return func


##### similarity measures #####

def sim(a, b, sig):  # perm. inv. similarity bewteen confs a and b
    simij = cdist(a, b, 'sqeuclidean')
    simij = np.exp(-simij / (4 * sig ** 2))
    sim = simij.sum() / ((2 * (np.pi * sig * sig) ** 0.5) ** D)
    return sim


def sim_cut(a, b, sig, r_c):  # perm. inv. similarity bewteen confs, w cutoff
    la = len(a)
    lb = len(b)
    amod = np.sqrt((a ** 2).sum(1))
    bmod = np.sqrt((b ** 2).sum(1))
    fi = f_cut(amod, r_c)
    fj = f_cut(bmod, r_c)
    fifj = np.outer(fi, fj)
    simij = cdist(a, b, 'sqeuclidean')
    simij = np.exp(-simij / (4 * sig ** 2)) * fifj
    sim = simij.sum() / (la * lb * (2 * (np.pi * sig * sig) ** 0.5) ** D)
    # print(max(amod), f_cut(max(amod), r_c))
    return sim


def sim_sq(a, b, sig, theta):  # squared similarity, used for debugging
    la = len(a)
    lb = len(b)

    sim = 0.
    for i in np.arange(la):
        for j in np.arange(lb):
            for l in np.arange(la):
                for m in np.arange(lb):
                    val = 0.
                    ri, rj = a[i], b[j]
                    rl, rm = a[l], b[m]
                    ris, rjs = ri.dot(ri), rj.dot(rj)
                    rls, rms = rl.dot(rl), rm.dot(rm)
                    rip, rjp = cart2pol(ri), cart2pol(rj)
                    rlp, rmp = cart2pol(rl), cart2pol(rm)
                    thetaij = rip[1] - rjp[1]
                    thetalm = rlp[1] - rmp[1]
                    C = np.exp(- (ris + rjs + rls + rms) / (4. * sig ** 2))
                    gamma_c = 1. / (2 * sig ** 2) * (np.dot(ri, rj) + np.dot(rl, rm))
                    gamma_s = 1. / (2 * sig ** 2) * (
                                np.sqrt(ris * rjs) * np.sin(thetaij) + np.sqrt(rls * rms) * np.sin(thetalm))
                    gamma = np.sqrt(gamma_c ** 2 + gamma_s ** 2)
                    thetaijlm = np.arctan2(gamma_s, gamma_c)  # np.arctan2(gamma_s**2,gamma_c**2)
                    val = C * np.exp(gamma * np.cos(thetaijlm + theta))
                    sim += val

    sim = sim / (la * lb * 4. * np.pi * sig ** 2) ** 2
    return sim


def inv_sim(a, b, sig):  # perm. inv. rot. inv. similarity bewteen confs.
    la = len(a)
    lb = len(b)

    amod = np.sqrt((a ** 2).sum(1))
    bmod = np.sqrt((b ** 2).sum(1))

    amod = np.atleast_2d(amod).T
    bmod = np.atleast_2d(bmod).T
    rimrjsq = cdist(amod, bmod, 'sqeuclidean')
    rirj = np.outer(amod, bmod)
    simij = np.exp(-(rimrjsq + 2 * rirj) / (4 * sig * sig)) * iv(0, rirj / (2 * sig * sig))
    sim = simij.sum() / (la * lb * (2 * (np.pi * sig * sig) ** 0.5) ** D)

    return sim


def sim_48(a, b, sig):  # perm. inv. O48 inv. similarity bewteen confs               
    syms = 48
    simo = 0.
    for sym in np.arange(syms):
        rm = rten48[sym]
        simo += sim(a, np.einsum('ik, jk -> ji', rm, b), sig)
    return simo * (1. / syms)


##### covariant kernels #####

def cov_sim(a, b, sig):  # SO(3) linear covariant kernel
    la = len(a)
    lb = len(b)
    sigsq = sig * sig
    ris, rjs = ((a ** 2).sum(1)), ((b ** 2).sum(1))
    ri, rj = np.sqrt(ris), np.sqrt(rjs)
    rirj = np.outer(ri, rj)
    risPrjs = ris[:, None] + rjs[:, None].T
    Cij = np.exp(-risPrjs / (4 * sigsq))
    gammaij = rirj / (2 * sigsq)
    r_c = 5.
    cutij = np.outer(f_cut(ri, r_c), f_cut(rj, r_c))
    Cij = Cij  # *cutij
    Iu2 = Cij * iv(1, gammaij)  # ((gammaij*np.cosh(gammaij) - np.sinh(gammaij))/gammaij**2)
    Iumat = np.zeros((la, lb, 3, 3))
    # Iumat[:, :, 0, 0] = Iu2
    # Iumat[:, :, 1, 1] = Iu2
    Iumat[:, :, 2, 2] = Iu2

    zaxis1, zaxis2 = np.zeros((la, 3)), np.zeros((lb, 3))
    zaxis1[:, 2], zaxis2[:, 2] = 1., 1.

    m1axis, m2axis = np.cross(a, zaxis1), np.cross(b, zaxis2)
    m1angle, m2angle = np.arccos(np.einsum('id, id -> i', a, zaxis1) / ri), np.arccos(
        np.einsum('id, id -> i', b, zaxis2) / rj)

    M1T = np.einsum('abi -> iba', tr.axangle2mat2(m1axis, m1angle))
    M2 = np.einsum('abi -> iab', tr.axangle2mat2(m2axis, m2angle))

    result = np.einsum('iab, ijbc , jcd -> ad', M1T, Iumat, M2)

    simo = result / ((2. * (np.pi * sigsq) ** 0.5) ** D)

    simo = simo

    if a is b:
        simo = (simo + simo.T) / 2.
    return simo


def cov_sim_SVD(a, b, sig):  # O(3) linear covariant kernel

    sigsq = sig * sig

    ris, rjs = ((a ** 2).sum(1)), ((b ** 2).sum(1))
    ri, rj = np.sqrt(ris), np.sqrt(rjs)

    risPrjs = ris[:, None] + rjs[:, None].T

    Cij = np.exp(-risPrjs / (4 * sigsq))

    Mji = np.einsum('ia, jb -> jiba', a, b)

    U, S, V = np.linalg.svd(Mji)

    D1ji = S[:, :, 0] / (2 * sigsq)

    Iu1 = np.cosh(D1ji)

    id_ji = np.zeros((U.shape))
    id_ji[:, :, 0, 0] = Iu1

    ref_ji = np.zeros((U.shape))
    ref_ji[:, :, 0, 0] = 0

    Iji = id_ji + ref_ji

    Iji = np.einsum('ij, jipq -> jipq', Cij, Iji)
    result = np.einsum('jiqp, jiqr, jisr -> ps', V, Iji, U)

    simo = result / ((2. * (np.pi * sigsq) ** 0.5) ** D)

    simo = simo

    if a is b:
        simo = (simo + simo.T) / 2.
    return simo


def cov_sim_tries(a, b, sig):  # SO(3) linear covariant kernel
    la = len(a)
    lb = len(b)
    sigsq = sig * sig
    ris, rjs = ((a ** 2).sum(1)), ((b ** 2).sum(1))
    ri, rj = np.sqrt(ris), np.sqrt(rjs)

    # take only particles within the first shell
    # cut = 1000.2
    # a, b = a[ri<= cut], b[rj <= cut]
    # ris, rjs = ris[ri<= cut], rjs[rj <= cut]
    # ri, rj = ri[ri<= cut], rj[rj <= cut]
    # print(len(a), len(b))

    rirj = np.outer(ri, rj)

    # Standard prior
    risPrjs = ris[:, None] + rjs[:, None].T
    Cij = np.exp(- risPrjs / (4 * sigsq))
    gammaij = rirj / (2 * sigsq)

    Iu1 = Cij * ((gammaij * np.cosh(gammaij) - np.sinh(
        gammaij)) / gammaij ** 2)  # iv(1, gammaij)#gammaij*(-0.5), iv(1, gammaij)
    # print(Iu1)
    an, bn = np.einsum('id, i -> id', a, 1. / ri), np.einsum('id, i -> id', b, 1. / rj)

    Iumat = np.einsum('ia, jb -> ijab', an, bn)

    result = np.einsum('ijab, ij -> ab', Iumat, Iu1)

    simo = result / ((2. * (np.pi * sigsq) ** 0.5) ** D)

    simo = simo

    if a is b:
        simo = (simo + simo.T) / 2.
    return simo


def cov_sim_tries2(a, b, sig):  # SO(3) linear covariant kernel
    la = len(a)
    lb = len(b)
    sigsq = sig * sig
    ris, rjs = ((a ** 2).sum(1)), ((b ** 2).sum(1))
    ri, rj = np.sqrt(ris), np.sqrt(rjs)
    rirj = np.outer(ri, rj)

    # ad-hoc prior
    risMrjs = ris[:, None] - rjs[:, None].T
    Cij = np.exp(- risMrjs ** 2 / (2. * sigsq))
    # Cij = np.exp(-np.sqrt(risMrjs**2)/np.sqrt(sigsq))
    # gammaij = rirj/(2*sigsq)
    # r_c = 3.
    # cutij = np.exp(gammaij)#gammaij**(-2)#(-gammaij/r_c + 1)
    Iu2 = Cij

    an, bn = np.einsum('id, i -> id', a, 1. / ri), np.einsum('id, i -> id', b, 1. / rj)

    Iumat = np.einsum('ia, jb -> ijab', an, bn)

    result = np.einsum('ijab, ij -> ab', Iumat, Iu2)

    simo = result / ((2. * (np.pi * sigsq) ** 0.5) ** D)

    simo = simo

    if a is b:
        simo = (simo + simo.T) / 2.
    return simo


def cov_sim_varsig(a, b, sig):  # SO(3) linear covariant kernel, variable sig
    la = len(a)
    lb = len(b)
    sigsq = sig * sig
    ris, rjs = ((a ** 2).sum(1)), ((b ** 2).sum(1))
    ri, rj = np.sqrt(ris), np.sqrt(rjs)
    sigis, sigjs = (0.15 * ri ** 2) ** 2, (0.15 * rj ** 2) ** 2
    sigisPsigjs = sigis[:, None] + sigjs[:, None].T
    sig_normij = 1. / (np.sqrt(sigisPsigjs / np.outer(sigis, sigjs))) ** D
    rirj = np.outer(ri, rj)
    risPrjs = ris[:, None] + rjs[:, None].T
    Cij = np.exp(-risPrjs / (4 * sigsq))
    gammaij = rirj / (2 * sigsq)
    r_c = 5.2
    cutij = np.outer(f_cut(ri, r_c), f_cut(rj, r_c))
    Cij = Cij * sig_normij  # *cutij
    Iu2 = Cij * ((gammaij * np.cosh(gammaij) - np.sinh(gammaij)) / gammaij ** 2)
    Iumat = np.zeros((la, lb, 3, 3))
    Iumat[:, :, 2, 2] = Iu2

    zaxis1, zaxis2 = np.zeros((la, 3)), np.zeros((lb, 3))
    zaxis1[:, 2], zaxis2[:, 2] = 1., 1.

    m1axis, m2axis = np.cross(a, zaxis1), np.cross(b, zaxis2)
    m1angle, m2angle = np.arccos(np.einsum('id, id -> i', a, zaxis1) / ri), np.arccos(
        np.einsum('id, id -> i', b, zaxis2) / rj)

    M1T = np.einsum('abi -> iba', tr.axangle2mat2(m1axis, m1angle))
    M2 = np.einsum('abi -> iab', tr.axangle2mat2(m2axis, m2angle))

    result = np.einsum('iab, ijbc , jcd -> ad', M1T, Iumat, M2)

    simo = result / (la * lb * (2. * np.pi) ** (0.5 * D))

    simo = simo
    if a is b:
        simo = (simo + simo.T) / 2.
    return simo


def cov_sim_sq(a, b, sig):  # SO(3) squared covariant kernel
    # definining variables and radial terms
    la = len(a)
    lb = len(b)
    sigsq = sig * sig
    ris, rjs = ((a ** 2).sum(1)), ((b ** 2).sum(1))
    ri, rj = np.sqrt(ris), np.sqrt(rjs)
    rirj = np.outer(ri, rj)
    risPrjs = ris[:, None] + rjs[:, None].T
    Cij = np.exp(-risPrjs / (4 * sigsq))
    gammaij = rirj / (2 * sigsq)
    Cijlm = np.outer(Cij, Cij)

    # finding the right transformations

    # matrices ri -> z and rj -> z
    zaxis1, zaxis2 = np.zeros((la, 3)), np.zeros((lb, 3))
    zaxis1[:, 2], zaxis2[:, 2] = 1., 1.
    mi_axis, mj_axis = np.cross(a, zaxis1), np.cross(b, zaxis2)
    mi_angle, mj_angle = np.arccos(np.einsum('id, id -> i', a, zaxis1) / ri), np.arccos(
        np.einsum('id, id -> i', b, zaxis2) / rj)

    Miz = np.einsum('abi -> iab', tr.axangle2mat2(mi_axis, mi_angle))
    Mjz = np.einsum('abi -> iab', tr.axangle2mat2(mj_axis, mj_angle))

    # rotating rl and rm and get matrices rl -> xz plane and rm -> xy plane

    ril_vecs, rjm_vecs = np.einsum('iab, lb -> ila', Miz, a), np.einsum('jab, mb -> jma', Mjz, b)
    mil_angle, mjm_angle = np.arctan2(ril_vecs[:, :, 0], ril_vecs[:, :, 1]), np.arctan2(rjm_vecs[:, :, 0],
                                                                                        rjm_vecs[:, :, 1])

    Milxz = np.einsum('abil -> ilab', tr.axangle2mat2(zaxis1, mil_angle))
    Mjmxz = np.einsum('abjm -> jmab', tr.axangle2mat2(zaxis2, mjm_angle))

    # multiply the obtained matrices to get Mil and Mjm
    Mil, Mjm = np.einsum('ilab, ibc -> ilac', Milxz, Miz), np.einsum('jmab, mbc -> jmac', Mjmxz, Mjz)

    # find the weight factor

    # rotate rl and rm onto xy plane

    # print(ril_vecs[0, 3, :], np.linalg.norm(ril_vecs[0, 3, :]))
    ril_vecs, rjm_vecs = np.einsum('ilab, ilb -> ila', Milxz, ril_vecs), np.einsum('jmab, jmb -> jma', Mjmxz, rjm_vecs)
    # print(ril_vecs[0, 3, :], np.linalg.norm(ril_vecs[0, 3, :])) # right until here
    # find optimal angle of rotation around y axis
    rlrm_costhetaijlm = np.einsum('ila, jma -> ijlm', ril_vecs, rjm_vecs)
    crossijlm = np.cross(ril_vecs[:, :, None, None, :], rjm_vecs[None, None, :, :, :])
    rlrm_sinthetaijlm = np.linalg.norm(crossijlm, axis=4)
    den_ijlm = rirj[:, None, :, None] + rlrm_sinthetaijlm
    alpha0_ijlm = np.arctan2(den_ijlm, rlrm_sinthetaijlm)
    # print(np.shape(alpha0_ijlm))   #reasonable until here

    # find best rotation around y axis
    yaxis = np.zeros((lb, 3))
    yaxis[:, 2] = 1.

    Rmax = np.einsum('abiljm -> ijlmab', tr.axangle2mat2(yaxis, alpha0_ijlm))

    # calculation of Hessian Matrix

    ij_rzl_rxm = np.outer(ril_vecs[:, :, ])

    # final result
    result = np.einsum('ilab, ijlmbc , jmcd -> ad', Mil, Rmax, Mjm)

    simo = result  # /(la*lb*(2.*(np.pi*sigsq)**0.5)**D)

    simo = simo  # /np.sqrt(sim_cut(a, a, sig, r_c)* sim_cut(b, b, sig, r_c))
    if a is b:
        simo = (simo + simo.T) / 2.
    return simo


def cov_sim_sq_SVD(a, b, sig):  # SO(3) squared covariant kernel
    la, lb = len(a), len(b)
    sigsq = sig * sig

    ris, rjs = ((a ** 2).sum(1)), ((b ** 2).sum(1))
    ri, rj = np.sqrt(ris), np.sqrt(rjs)

    # take only particles within the first shells
    cut = 6.0  ### USE A LOW VALUE FOR DEBUGGING
    a, b = a[ri <= cut], b[rj <= cut]
    ris, rjs = ris[ri <= cut], rjs[rj <= cut]
    ri, rj = ri[ri <= cut], rj[rj <= cut]
    r_c = cut / 2.0
    # print(len(a), len(b))

    riPrj = np.add.outer(ris, rjs)
    risPrjsPrlsPrms = np.add.outer(riPrj, riPrj)

    Cijlm = np.exp(- risPrjsPrlsPrms / (4 * sigsq))

    ab = np.einsum('ia, jb -> jiba', a, b)  # note: need "ji" because Tr(iRj)=Tr(Rji)

    Mjiml = ab[:, :, None, None, :, :] + ab[:, :, :, :]

    U, S, V = np.linalg.svd(Mjiml)  # note: M = U S V

    D1jiml = np.sum(S, axis=4) / (2. * sigsq)
    D2jiml = (S[:, :, :, :, 0] - S[:, :, :, :, 1]) / (2. * sigsq)

    # Exact integration: iv(1, D1jiml)
    # Plain Maximum: np.exp(D1jiml)
    # Laplace approximation 1: np.exp((D1jiml+1/(2*D1jiml)))/np.sqrt(2*np.pi*D1jiml)
    # Laplace approximation 2: np.exp(D1jiml)/np.sqrt(2*np.pi*(D1jiml+1))

    Besseljiml1 = iv(1, D1jiml)
    Besseljiml2 = iv(1, D2jiml)

    id_jiml = np.zeros((U.shape))
    id_jiml[:, :, :, :, 0, 0] = Besseljiml1
    id_jiml[:, :, :, :, 1, 1] = Besseljiml1

    ref_jiml = np.zeros((U.shape))
    ref_jiml[:, :, :, :, 0, 0] = Besseljiml2
    ref_jiml[:, :, :, :, 1, 1] = - Besseljiml2

    Ijiml = id_jiml + ref_jiml
    # Ijiml = np.einsum('ijlm, jimlpq -> jimlpq', Cijlm, Ijiml)

    ### Use this to use only sensible triplets, NOT WORKING ###
    disti = np.array(cdist(a, a, 'euclidean'))  # Matrix of distances within a, shape la*la
    distj = np.array(cdist(b, b, 'euclidean'))  # Matrix of distances within b, shape lb*lb
    cuta = tripl_cut((np.reshape(np.tile(ri, la), (la, la))), (np.reshape(np.repeat(ri, la), (la, la))), disti,
                     r_c)  # Function that is 1 if a triplet is sensible in A
    cutb = tripl_cut((np.reshape(np.tile(rj, lb), (lb, lb))), (np.reshape(np.repeat(rj, lb), (lb, lb))), distj,
                     r_c)  # Function that is 1 if a triplet is sensible in B
    Ijiml = np.einsum('ijlm, jimlpq, il, jm -> jimlpq', Cijlm, Ijiml, cuta, cutb)
    ####

    simo = np.einsum('jimlqp, jimlqr, jimlsr -> ps', V, Ijiml, U)  # effectively V^T, I, U^T

    L = ((2 * (np.pi * sigsq) ** 0.5) ** D) ** 2

    simo = simo / L / 2.

    return simo


def cov_sim_theano(a, b,
                   sig):  ### Kernel, calculated using THEANO, that considers valid triplets the ones which have all distances smaller than cut
    cut = 4.31
    theta = 0.4  # Decay length of the cutoff function

    if (cut < 100.0):
        ### Take only particles within the ficutoff ###
        ri, rk = np.sqrt((a ** 2).sum(1)), np.sqrt((b ** 2).sum(1))  # Distances

        a, b = a[ri < cut], b[rk < cut]

    # print(len(a))
    return Hessian3c([0, 0, 0], [0, 0, 0], a, b, sig, cut, theta)


def theano_exp(a, b, sig,
               gamma):  ### Kernel, calculated using THEANO, that considers valid triplets the ones which have all distances smaller than cut
    cut = 4.31
    theta = 0.4
    twogammasq = 2.0 * gamma * gamma
    ### Take only particles within the cutoff ###
    ri, rk = np.sqrt((a ** 2).sum(1)), np.sqrt((b ** 2).sum(1))  # Distances

    a, b = a[ri < cut], b[rk < cut]

    return Hessian3e([0, 0, 0], [0, 0, 0], a, b, sig, cut, theta, twogammasq)


def square_3b(a, b, sig):
    cut = 4.31
    theta = 0.4  # Decay length of the cutoff function

    if (cut < 100.0):
        ### Take only particles within the ficutoff ###
        ri, rk = np.sqrt((a ** 2).sum(1)), np.sqrt((b ** 2).sum(1))  # Distances

        a, b = a[ri < cut], b[rk < cut]

    return Hessian5c([0, 0, 0], [0, 0, 0], a, b, sig, cut, theta)


def energy_square(a, b, sig):
    cut = 4.31
    theta = 0.4
    if (cut < 100.0):
        ### Take only particles within the cutoff ###
        ri, rk = np.sqrt((a ** 2).sum(1)), np.sqrt((b ** 2).sum(1))  # Distances
        a, b = a[ri < cut], b[rk < cut]

    return Grad3c([0, 0, 0], [0, 0, 0], a, b, sig, cut, theta)


def energy_exp(a, b, sig, gamma):
    cut = 4.5
    theta = 0.4
    twogammasq = 2.0 * gamma * gamma
    ### Take only particles within the cutoff ###
    ri, rk = np.sqrt((a ** 2).sum(1)), np.sqrt((b ** 2).sum(1))  # Distances

    a, b = a[ri < cut], b[rk < cut]
    return Grad3c_exp([0, 0, 0], [0, 0, 0], a, b, sig, cut, theta, twogammasq)


def squared_exp(a, b, sig, gamma):
    cut = 4.5
    theta = 0.4
    twogammasq = 2.0 * gamma * gamma
    ### Take only particles within the cutoff ###
    ri, rk = np.sqrt((a ** 2).sum(1)), np.sqrt((b ** 2).sum(1))  # Distances

    a, b = a[ri < cut], b[rk < cut]

    expab = Hessian3e([0, 0, 0], [0, 0, 0], a, b, sig, cut, theta, twogammasq)
    expaa = Hessian3e([0, 0, 0], [0, 0, 0], a, a, sig, cut, theta, twogammasq)
    expbb = Hessian3e([0, 0, 0], [0, 0, 0], b, b, sig, cut, theta, twogammasq)

    return (expab / np.sqrt(expaa * expbb))


def cov_sim48(a, b, sig, gamma, cut=4.0):  # O48 covariant kernel
    cut = 100.0
    ris, rjs = ((a ** 2).sum(1)), ((b ** 2).sum(1))
    ri, rj = np.sqrt(ris), np.sqrt(rjs)

    # take only particles within the first shells
    # a, b = a[ri<= cut], b[rj <= cut]
    # la, lb = len(a), len(b)

    simgij = np.array([cdist(a, np.einsum('ik, jk -> ji', rten48[sym], b), 'sqeuclidean') for sym in np.arange(48)])
    simgij = np.exp(-simgij / (4 * sig ** 2))
    simg = np.einsum('gij -> g', simgij) / ((2 * (np.pi * sig * sig) ** 0.5) ** D)
    # exp:np.exp((2*simg - 2)/(2*theta))
    # exp:np.exp((2*simg - 2)/(2*theta))
    # simsq: simg**2
    simo = np.einsum(' kij , k -> ij ', rten48, np.exp(simg / (2 * gamma ** 2)))

    return simo * (1. / 48.)


def cov_sim240(a, b, sig, gamma, cut=4.0):  # O48 covariant kernel
    cut = 100.0
    ris, rjs = ((a ** 2).sum(1)), ((b ** 2).sum(1))
    ri, rj = np.sqrt(ris), np.sqrt(rjs)

    simgij = np.array(
        [cdist(a, np.einsum('ik, jk -> ji', rot240[sym], b), 'sqeuclidean') for sym in np.arange(len(rot240))])
    simgij = np.exp(-simgij / (4 * sig ** 2))
    simg = np.einsum('gij -> g', simgij) / ((2 * (np.pi * sig * sig) ** 0.5) ** D)
    simo = np.einsum(' kij , k -> ij ', rot240, np.exp(simg / (2 * gamma ** 2)))

    return simo * (1. / 120.)


def exp_noncov(a, b, sig, theta):  # Exponential non covariant kernel
    rot0 = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
    ris, rjs = ((a ** 2).sum(1)), ((b ** 2).sum(1))
    ri, rj = np.sqrt(ris), np.sqrt(rjs)
    # take only particles within the first shells
    cut = 2.7  # 4.17
    a, b = a[ri <= cut], b[rj <= cut]
    la, lb = len(a), len(b)

    simgij = np.array(cdist(a, b, 'sqeuclidean'))
    simgij = np.exp(-simgij / (4 * sig ** 2))
    simg = np.einsum('ij -> ', simgij) / ((2 * (np.pi * sig * sig) ** 0.5) ** D)
    simo = np.einsum(' ij ,  -> ij ', rot0, simg ** 2)
    return simo


##### feature vectors #####

def LJ_force(a, rm, eps):  # LJ force

    ds = np.sqrt((a ** 2).sum(1))
    term = 12 * eps / ds * ((rm / ds) ** 12 - (rm / ds) ** 6) / ds

    f = np.einsum("i, id -> d", term, a)
    return f


def LJ_force_params(a, rms, epss):  # LJ over grid of parameters

    ds = np.sqrt((a ** 2).sum(1))
    epsds = np.outer(epss, 1. / ds ** 2)
    rmds = np.outer(rms, 1. / ds)

    term = 12 * epsds * ((rmds) ** 12 - (rmds) ** 6)

    f = np.einsum("ji, id -> jd", term, a)
    return f


def harmonic_force(a, rm, amp):  # harmonic force

    ds = np.sqrt((a ** 2).sum(1))
    ds[ds > 1.5] = rm
    term = amp * (ds - rm) / ds
    f = np.einsum("i, id -> d", term, a)
    return f


def integr_LJs_mat_modegen(a, b, eps_mod, rm_mod, sig_eps, sig_rm):  # LJ with integrated mode

    logr_m = np.log(rm_mod)

    amod = np.sqrt((a ** 2).sum(1))
    bmod = np.sqrt((b ** 2).sum(1))
    outer = np.outer(amod, bmod)
    term24 = np.exp(24. * logr_m + 312. * sig_rm ** 2) / (outer ** 12)
    term18 = - np.exp(18. * logr_m + 180. * sig_rm ** 2) * (
                1. / np.outer(amod ** 6, bmod ** 12) + 1. / np.outer(amod ** 12, bmod ** 6))
    term12 = np.exp(12. * logr_m + 84. * sig_rm ** 2) / (outer ** 6)
    terms = (term24 + term18 + term12) / outer ** 2
    rrt = np.einsum("ia, jb -> iajb", a, b)
    mat = np.einsum("ij,iajb->ab", terms, rrt)

    mat = mat * 144. * np.exp(2 * np.log(eps_mod) + 4 * sig_eps ** 2)
    return mat


def integr_LJs_mat_meangen(a, b, rm_mean, eps_mean, sig_rm, sig_eps):  # LJ with integrated mean

    logr_m = np.log(rm_mean)
    amod = np.sqrt((a ** 2).sum(1))
    bmod = np.sqrt((b ** 2).sum(1))
    outer = np.outer(amod, bmod)
    term24 = np.exp(24. * logr_m + 276. * sig_rm ** 2) / outer ** 12
    term18 = - np.exp(18. * logr_m + 153. * sig_rm ** 2) * (
                1. / np.outer(amod ** 6, bmod ** 12) + 1. / np.outer(amod ** 12, bmod ** 6))
    term12 = np.exp(12. * logr_m + 66. * sig_rm ** 2) / outer ** 6
    terms = (term24 + term18 + term12) / outer ** 2
    rrt = np.einsum("ia, jb -> iajb", a, b)

    mat = np.einsum("ij,iajb->ab", terms, rrt)
    mat = mat * 144. * np.exp(2. * np.log(eps_mean) + sig_eps ** 2)
    if a is b:
        mat = (mat + mat.T) / 2.
    return mat


def LJs_mat(a, b, rm, eps):
    amod = np.sqrt((a ** 2).sum(1))
    bmod = np.sqrt((b ** 2).sum(1))
    outer = np.outer(amod, bmod)
    term24 = rm ** 24 / outer ** 12
    term18 = - rm ** 18 * (1. / np.outer(amod ** 12, bmod ** 6) + 1. / np.outer(amod ** 6, bmod ** 12))
    term12 = rm ** 12 / outer ** 6
    terms = (term24 + term18 + term12) / outer ** 2
    rrt = np.einsum("ia, jb -> iajb", a, b)
    mat = np.einsum("ij,iajb->ab", terms, rrt)
    mat = 144. * eps * eps * mat
    if (a == b).all():
        mat = (mat + mat.T) / 2.
    return mat


def SW_mat(a, b, eq_ang, eps2):
    amod = np.sqrt((a ** 2).sum(1))
    bmod = np.sqrt((b ** 2).sum(1))
    outera = np.outer(amod, amod)
    outerb = np.outer(bmod, bmod)
    cosjk = np.einsum('jd, kd -> jk', a, a) / outera
    coslm = np.einsum('jd, kd -> jk', b, b) / outerb
    term1jklm = np.einsum('jk, lm -> jklm', cosjk, coslm)
    term2jklm = -1. / 3. * (cosjk[:, :, None, None] + coslm[:, :, None, None].T)
    term3 = np.array((1. / 3.) ** 2)
    a_sc = np.einsum('jdf, jf -> jd', a[:, :, None], 1. / amod[:, None] ** 2)
    b_sc = np.einsum('jdf, jf -> jd', b[:, :, None], 1. / bmod[:, None] ** 2)
    Cjk1 = -np.einsum('jdkf, jkf -> jdk', (a[:, :, None] + a[:, :, None].T)[:, :, :, None], 1. / outera[:, :, None])
    Cjk2 = np.einsum('jkf, jdkf -> jdk', cosjk[:, :, None], (a_sc[:, :, None] + a_sc[:, :, None].T)[:, :, :, None])
    Clm1 = -np.einsum('jdkf, jkf -> jdk', (b[:, :, None] + b[:, :, None].T)[:, :, :, None], 1. / outerb[:, :, None])
    Clm2 = np.einsum('jkf, jdkf -> jdk', coslm[:, :, None], (b_sc[:, :, None] + b_sc[:, :, None].T)[:, :, :, None])
    Cjklmab = np.einsum('jak, lbm ->jklmab', Cjk1 + Cjk2, Clm1 + Clm2)
    result = np.einsum('jklmab , jklm -> ab', Cjklmab, term1jklm + term2jklm + term3[None, None, None, None])
    return result


def integr_harmonic_mat(a, b, sig_amp, sig_r0):
    mat = np.zeros((D, D))
    d_cut = 1.5
    for i in np.arange(len(a)):
        for j in np.arange(len(b)):
            ri, rj = a[i], b[j]
            di, dj = np.sqrt(ri.dot(ri)), np.sqrt(rj.dot(rj))
            if di <= d_cut and dj <= d_cut:
                term = (di * dj + np.exp(sig_r0 ** 2) - di - dj)
                mat += term * np.outer(ri, rj) / (di * dj)
    mat = mat * np.exp(np.log(72 * 2) + sig_amp ** 2)
    return mat


def COM(a):
    com = a.sum(0)
    return com


def m1_kSE(a, sig, theta):
    amod = np.sqrt((a ** 2).sum(1))
    sigsq, thetasq = sig * sig, theta * theta
    term = np.exp(-amod ** 2 / (2 * (sigsq + thetasq))) * thetasq / (sigsq + thetasq)
    vec = np.einsum('i, id -> d', term, a)
    return vec


def m2_kSE(a, sig, theta):
    amod = np.sqrt((a ** 2).sum(1))
    sigsq, thetasq = sig * sig, theta * theta
    term = np.exp(-cdist(a, a, 'sqeuclidean') / (2 * (2 * sigsq + thetasq))) * thetasq / (
                2 * sigsq + thetasq)  # *(r1 + r2)
    r1pr2 = a[:, :, None] + a[:, :, None].T
    vec = np.einsum('ij, idj -> d', term, r1pr2)
    return vec


def repulsion(a, m_theta):
    ri = np.linalg.norm(a, axis=1)
    ahat = np.einsum('nd, n -> nd', a, 1 / ri)
    Ci = - (m_theta / (ri)) ** (-13)
    result = np.einsum('i, id -> d ', Ci, ahat)
    return result


def repulsion_energy(a, m_theta):
    ri = np.linalg.norm(a, axis=1)
    Ci = (m_theta * 12.0 / (ri)) ** (-12)
    result = np.sum(Ci)
    return result


##### gaussian process module #####

class GaussianProcess3:

    def __init__(self, ker=['sim', 'LI_MAT'], fvecs=['cart', 'LJs'], theta0=[1e-1], m_theta0=[None],
                 nugget=1000. * MACHINE_EPSILON, sig=0.5, gamma=1.0, optimizer="fmin_l_bfgs_b", \
                 bounds=(0.1, 10), calc_error=False, eval_grad=False):
        self.sig = sig
        self.theta0 = theta0
        self.gamma = gamma
        self.ker = ker
        self.fvecs = fvecs
        self.nugget = nugget
        self.optimizer = optimizer
        self.bounds = bounds
        self.calc_error = calc_error
        self.eval_grad = eval_grad
        self.r_c = 5.  # 3.75
        self.m_theta0 = m_theta0

    def inv_ker(self, a, b):
        theta0 = self.stheta0
        sig = self.sig
        ker = self.sker
        eval_grad = self.eval_grad

        if ker == 'sim':
            simscaled = (sim(a, b, sig))
            result = (simscaled) ** theta0
            if eval_grad:
                grad = result * np.log(simscaled)
                return result, grad
            else:
                return (result)
        elif ker == 'SE':
            distsq = - 2 * sim(a, b, sig) + 2.  # sim(a, a, sig) + sim(b, b, sig)
            result = np.exp(-distsq / (2. * theta0))
            if eval_grad:
                grad = result * distsq / (theta0 ** 2)
                return result, grad
            else:
                return result
        elif ker == 'sim_cut':
            r_c = self.r_c
            result = (sim_cut(a, b, sig, r_c) / np.sqrt(sim_cut(a, a, sig, r_c) * sim_cut(b, b, sig, r_c)))  # **2
            return result
        elif ker is 'inv_sim':
            result = (inv_sim(a, b, sig) / np.sqrt(inv_sim(a, a, sig) * inv_sim(b, b, sig))) ** 2

            return result
        elif ker is 'sim_sq':
            result = sim_sq(a, b, sig, 0)

            return result
        elif ker == 'inv_sim_sq':
            result = (inv_sim_sq(a, b, sig))  # /np.sqrt(inv_sim_sq(a, a, sig)* inv_sim_sq(b, b, sig)))**2
            return result
        elif ker is "LI_MAT":
            v1s, v2s = self.v1s, self.v2s
            v1s_n, v2s_n = normalize(v1s, axis=0), normalize(v2s, axis=0)
            a_mat = v1s.dot(v1s_n.T)
            b_mat = v2s.dot(v2s_n.T)
            d_sqrd = 1. / self.LL * np.sum((a_mat - b_mat) ** 2)
            result = np.exp(-d_sqrd / (2 * theta0))
            if eval_grad:
                grad = result * d_sqrd / (theta0 ** 2)
                return result, grad

            else:
                return result
        elif ker == 'id':
            return 1.
        elif ker is "INERTIA_TEN":
            a_mat = a.dot(a.T)
            b_mat = b.dot(b.T)
            d_sqrd = 1. / self.LL * np.sum((a_mat - b_mat) ** 2)
            result = np.exp(-d_sqrd / (2 * theta0))
            # print(result)
            return result

        else:
            print("Correlation model %s not understood" % ker)
            return None

    def feat_vecs(self, a, b):
        fvecs = self.sfvecs

        if fvecs == 'cart':

            self.LL = D
            f_vecsa, f_vecsb = np.zeros((self.LL, D)), np.zeros((self.LL, D))
            fvs = np.identity(D)
            for i in np.arange(D):
                f_vecsa[i], f_vecsb[i] = fvs[i], fvs[i]
            fvec_ten = np.einsum('ia, jb -> iajb', f_vecsa, f_vecsb)
        elif fvecs is 'gen':
            sig = self.sig
            theta0 = self.stheta0
            self.LL = 3
            f_vecsa, f_vecsb = np.zeros((self.LL, D)), np.zeros((self.LL, D))
            f_vecsa[0], f_vecsa[1], f_vecsa[2] = COM(a), m1_kSE(a, sig, theta0), m2_kSE(a, sig, theta0)
            f_vecsb[0], f_vecsb[1], f_vecsb[2] = COM(b), m1_kSE(b, sig, theta0), m2_kSE(b, sig, theta0)
            fvec_ten = np.einsum('ia, jb -> iajb', f_vecsa, f_vecsb)
        elif fvecs == 'LJ':
            self.LL = 1
            f_vecsa, f_vecsb = LJ_force(a, 2.3, 1.)[None, :], LJ_force(b, 2.3, 1.)[None, :]
            fvec_ten = np.einsum('ia, jb -> iajb', f_vecsa, f_vecsb)
        elif fvecs is 'LJs':
            self.LL = 25
            f_vecsa, f_vecsb = np.zeros((self.LL, D)), np.zeros((self.LL, D))

            epss, rms = np.linspace(1., 2., 5), np.linspace(2., 2.3, 5)
            rmeps = np.dstack(np.meshgrid(rms, epss)).reshape(-1, 2)
            f_vecsa, f_vecsb = LJ_force_params(a, rmeps[:, 0], rmeps[:, 1]), LJ_force_params(b, rmeps[:, 0],
                                                                                             rmeps[:, 1])
            self.v1s, self.v2s = f_vecsa, f_vecsb
            fvec_ten = np.einsum('ia, jb -> iajb', f_vecsa, f_vecsb)
        elif fvecs is 'LJ_int':
            self.LL = 1
            lj_mat = integr_LJs_mat_meangen(a, b, 2.8, 1., 0.02, 0.02)
            fvec_ten = lj_mat[None, :, None, :]
        elif fvecs == 'EAM':
            self.LL = 1

            at1 = Atoms('Fe' * (len(a) + 1), positions=np.vstack((np.array([0, 0, 0]), a)))
            at2 = Atoms('Fe' * (len(b) + 1), positions=np.vstack((np.array([0, 0, 0]), b)))
            at1.set_calculator(calc), at2.set_calculator(calc)
            f_vecsa, f_vecsb = at1.get_forces()[0][None, :], at2.get_forces()[0][None, :]
            fvec_ten = np.einsum('ia, jb -> iajb', f_vecsa, f_vecsb)

        elif fvecs == '2b':
            self.LL = 1
            optimizer = self.optimizer
            sig = self.sig
            if optimizer:
                theta0 = self.theta0
                lj_mat = cov_sim_tries(a, b, theta0[0])
            else:
                lj_mat = cov_sim_tries(a, b, sig)

            fvec_ten = lj_mat[None, :, None, :]
        elif fvecs == 'cov_sim_sq':
            self.LL = 1
            sig = self.sig
            lj_mat = cov_sim_sq_SVD(a, b, sig)
            fvec_ten = lj_mat[None, :, None, :]
        elif fvecs == 'cov_sim_sq_cons':
            self.LL = 1
            sig = self.sig
            lj_mat = cov_sim_sq_cons(a, b, sig)
            fvec_ten = lj_mat[None, :, None, :]
        elif fvecs == 'cov_sim_sq_cons2':
            self.LL = 1
            sig = self.sig
            lj_mat = cov_sim_sq_cons2(a, b, sig)
            fvec_ten = lj_mat[None, :, None, :]
        elif fvecs == 'cov_sim_sq_cons3':
            self.LL = 1
            sig = self.sig
            lj_mat = cov_sim_sq_cons3(a, b, sig)
            fvec_ten = lj_mat[None, :, None, :]

        elif fvecs == 'cov_sim_sq_cons4':
            self.LL = 1
            sig = self.sig
            lj_mat = cov_sim_sq_cons4(a, b, sig)
            fvec_ten = lj_mat[None, :, None, :]

        elif fvecs == '3b':
            self.LL = 1
            sig = self.sig
            opt = self.optimizer
            theta0 = self.theta0
            if opt:
                lj_mat = cov_sim_theano(a, b, theta0[0])
            else:
                lj_mat = cov_sim_theano(a, b, sig)
            fvec_ten = lj_mat[None, :, None, :]

        elif fvecs == 'square_3b':
            self.LL = 1
            sig = self.sig
            opt = self.optimizer
            theta0 = self.theta0
            if opt:
                lj_mat = square_3b(a, b, theta0[0])
            else:
                lj_mat = square_3b(a, b, sig)
            fvec_ten = lj_mat[None, :, None, :]

        elif fvecs == 'theano_exp':
            self.LL = 1
            sig = self.sig
            gamma = self.gamma
            lj_mat = theano_exp(a, b, sig, gamma)
            fvec_ten = lj_mat[None, :, None, :]

        elif fvecs == 'mb':
            self.LL = 1
            sig = self.sig
            gamma = self.gamma
            theta0 = self.theta0
            lj_mat = cov_sim48(a, b, sig, gamma)
            fvec_ten = lj_mat[None, :, None, :]

        elif fvecs == 'cov_sim_240':
            self.LL = 1
            sig = self.sig
            gamma = self.gamma
            theta0 = self.theta0
            lj_mat = cov_sim240(a, b, sig, gamma)
            fvec_ten = lj_mat[None, :, None, :]

        elif fvecs == 'exp_noncov':
            self.LL = 1
            sig = self.sig
            theta0 = self.theta0
            lj_mat = exp_noncov(a, b, sig, theta0)
            fvec_ten = lj_mat[None, :, None, :]

        elif fvecs == 'squared_exp':
            self.LL = 1
            sig = self.sig
            gamma = self.gamma
            lj_mat = squared_exp(a, b, sig, gamma)
            fvec_ten = lj_mat[None, :, None, :]

        elif fvecs is 'harm':  # note LJ equilibrium amplitude is 72
            self.LL = 9
            f_vecsa, f_vecsb = np.zeros((self.LL, D)), np.zeros((self.LL, D))
            idx = 0
            for amp in np.linspace(62, 82, 3):
                for rm in np.linspace(0.9, 1.1, 3):
                    f_vecsa[idx] = harmonic_force(a, rm, amp)
                    f_vecsb[idx] = harmonic_force(b, rm, amp)
                    idx += 1
            fvec_ten = np.einsum('ia, jb -> iajb', f_vecsa, f_vecsb)

        return fvec_ten

    def mat_kernel_func(self, a, b):
        """
        Calculation of the matrix valued kernel function.
        
        Parameters
        ----------

        
        Returns
        -------
        K(a, b): matrix valued kernel function 
        """

        sig = self.sig
        theta0 = self.theta0
        gamma = self.gamma
        ker = self.ker
        feat_vecs = self.feat_vecs
        inv_ker = self.inv_ker
        fvecs = self.fvecs
        K = np.zeros((D, D))
        eval_grad = self.eval_grad

        if eval_grad:
            K_g = np.zeros((len(theta0), D, D))
            for s in np.arange(len(ker)):
                self.sker = ker[s]
                self.sfvecs = fvecs[s]
                self.stheta0 = theta0[s]
                fvec_ten = feat_vecs(a, b)
                self.fvec_ten = fvec_ten
                k_inv, k_inv_g = inv_ker(a, b)
                d_inv, d_inv_g = np.ones(self.LL) * k_inv, np.ones(self.LL) * k_inv_g
                K += np.einsum('iaib, i -> ab', fvec_ten, d_inv)
                K_g[s] += np.einsum('iaib, i -> ab', fvec_ten, d_inv_g)

            return K, K_g
        else:
            for s in np.arange(len(ker)):
                self.sker = ker[s]
                self.sfvecs = fvecs[s]
                self.stheta0 = theta0[s]
                fvec_ten = feat_vecs(a, b)
                self.fvec_ten = fvec_ten
                k_inv = np.ones(self.LL) * inv_ker(a, b)
                K += np.einsum('iaib, i -> ab', fvec_ten, k_inv)

            return K

    def calc_kernel_matrix(self, X, sig, theta0):
        """
        Calculation of the Gram Matrix.
        
        Parameters
        ----------
        X : array with shape (n_samples, n_features)
        
        Returns
        -------
        K: the Gram Matrix (covariance matrix) of the data
        """
        self.theta0 = theta0
        ker = self.ker
        mat_kernel_func = self.mat_kernel_func
        Ntrain = self.Ntrain
        diag = np.identity(Ntrain * D) * self.nugget
        off_diag = np.zeros((Ntrain * D, Ntrain * D))
        eval_grad = self.eval_grad

        if eval_grad:
            g_diag = np.zeros((len(theta0), Ntrain * D, Ntrain * D))
            g_off_diag = np.zeros((len(theta0), Ntrain * D, Ntrain * D))
            for i in np.arange(Ntrain):
                k, kg = mat_kernel_func(X[i], X[i])
                diag[D * i:D * i + D, D * i:D * i + D] += k
                g_diag[:, D * i:D * i + D, D * i:D * i + D] += kg + self.nugget
                for j in np.arange(i):
                    off_diag[D * i:D * i + D, D * j:D * j + D], g_off_diag[:, D * i:D * i + D,
                                                                D * j:D * j + D] = mat_kernel_func(X[i], X[j])
            K = diag + off_diag + off_diag.T
            K_g = g_diag + g_off_diag + np.transpose(g_off_diag, (0, 2, 1))
            self.K = K
            # print("Is K symmetric? ", (K == K.T).all())
            # print("Is K positive definite?  ", (np.linalg.eigvalsh(K)> 0).all())
            return K, K_g

        else:
            for i in np.arange(Ntrain):
                diag[D * i:D * i + D, D * i:D * i + D] += mat_kernel_func(X[i], X[i])
                for j in np.arange(i):
                    off_diag[D * i:D * i + D, D * j:D * j + D] = mat_kernel_func(X[i], X[j])
            K = diag + off_diag + off_diag.T
            np.set_printoptions(precision=3)
            self.K = K
            # print("Is K symmetric? ", (K == K.T).all())
            # eigs = np.linalg.eigvalsh(K)
            # print("Is K positive definite?  ", (eigs> 0).all())

            return K

    def get_gram(self):
        K = self.K
        return K

    def log_marginal_likelihood(self, sig, theta0):

        Xtrain = self.Xtrain
        Xtrain = self.Xtrain
        eval_grad = self.eval_grad

        if eval_grad:
            K, K_g = self.calc_kernel_matrix(Xtrain, sig, theta0)
        else:
            K = self.calc_kernel_matrix(Xtrain, sig, theta0)

        L = cholesky(K, lower=True)
        alpha = cho_solve((L, True), self.ytrain)

        log_likelihood = -0.5 * np.dot(self.ytrain.T, alpha)
        log_likelihood -= np.log(np.diag(L)).sum()
        log_likelihood -= K.shape[0] / 2. * np.log(2. * np.pi)

        if eval_grad:
            tmp = np.outer(alpha, alpha)
            tmp -= cho_solve((L, True), np.eye(K.shape[0]))
            log_likelihood_g = 0.5 * np.einsum('lj , slj -> s', tmp, K_g)
            return log_likelihood, log_likelihood_g
        else:
            return log_likelihood

    def constrained_optimization(self, obj_func, initial_theta, bounds):
        approx = not self.eval_grad
        if self.optimizer == "fmin_l_bfgs_b":
            theta0_opt, func_min, convergence_dict = fmin_l_bfgs_b(obj_func, initial_theta, bounds=bounds,
                                                                   approx_grad=approx, iprint=1)
            print(theta0_opt)
            print("fmin_l_bfgs_b theta0_opt and func_min are ", theta0_opt, func_min, convergence_dict)
        elif self.optimizer == "Nelder-Mead":
            optimum = minimize(fun=obj_func, x0=initial_theta, args=(), method="Nelder-Mead", bounds=bounds,
                               options={'disp': True})
            print(optimum)
            theta0_opt, func_min = optimum.x, optimum.fun
            print("Nelder-Mead theta0_opt and func_min are ", theta0_opt, func_min, optimum.message)
        return theta0_opt, func_min

    def fit(self, X=None, y_ten=None):
        """
        The Gaussian Process model fitting method.

        Parameters
        ----------
        X : array with shape (n_samples, n_features)

        y : array with shape (n_samples)
        
        Returns
        -------
        gp : self
            A fitted Gaussian Process model object awaiting data to perform
            predictions.
        """
        self.Xtrain = X
        self.Ntrain = len(X)
        y = np.resize(y_ten, (len(y_ten) * D, 1))
        self.ytrain = y
        eval_grad = self.eval_grad

        if self.m_theta0[0] is not None:
            m_theta = self.m_theta0[0]
            mvec = np.array([repulsion(X[n], m_theta) for n in np.arange(len(X))])
            y_ten = y_ten - mvec

        if self.optimizer is not None:

            def obj_func(theta0):
                if eval_grad:
                    lml, lml_grad = self.log_marginal_likelihood(self.sig, theta0)
                    print("D: ", theta0, -lml, -lml_grad)
                    return -lml, -lml_grad
                else:
                    lml = self.log_marginal_likelihood(self.sig, theta0)
                    print("D: ", theta0, -lml)
                    return -lml

            # First optimize starting from theta specified in kernel
            initial_theta = self.theta0
            theta_opt, func_min = self.constrained_optimization(obj_func, initial_theta, self.bounds)
            self.theta0 = theta_opt
            self.log_marginal_likelihood_value = -func_min

        sig = self.sig
        theta0 = self.theta0
        calc_kernel_matrix = self.calc_kernel_matrix
        self.eval_grad = False
        K = calc_kernel_matrix(X, sig, theta0)
        # print('K gram matrix is: \n', K)

        # invert covariance matrix
        try:
            inv = LA.pinv2(K)
        except LA.LinAlgError as err:
            print("pinv2 failed: %s. Switching to pinvh" % err)
            try:
                inv = LA.pinvh(K)
            except LA.LinAlgError as err:
                print("pinvh failed: %s. Switching to pinv2" % err)
                inv = None

        # alpha is the vector of regression coefficients of GaussianProcess
        inv_ten = np.reshape(inv, (self.Ntrain, D, self.Ntrain, D))
        self.inv_ten = inv_ten
        self.K = K
        self.alpha_ten = np.einsum('ndND, ND -> nd ', inv_ten, y_ten)

        # second method, cholesky decomposition
        """
        self.L = cholesky(K, lower = True)
        self.alpha2 = cho_solve((self.L, True), self.ytrain)
        idx = np.argmax(np.abs(self.alpha-self.alpha2))
        
        print(self.alpha[idx],self.alpha2[idx], np.abs(self.alpha-self.alpha2)[idx],np.max(np.abs(self.alpha-self.alpha2)),np.mean(np.abs(self.alpha)),np.mean(np.abs(self.alpha2)),np.mean(np.abs(self.alpha-self.alpha2)), )
        """

    def predict(self, X):
        """
        This function evaluates the Gaussian Process model at X.
    
        Returns
        -------
        y : array_like
        """

        Xtrain = self.Xtrain
        Ntrain = self.Ntrain
        self.X = X
        calc_error = self.calc_error
        K2 = np.zeros((len(X) * D, Ntrain * D))
        K2_ten = np.zeros((len(X), Ntrain, D, D))
        ker = self.ker
        theta0 = self.theta0
        sig = self.sig
        nugget = self.nugget
        mat_kernel_func = self.mat_kernel_func
        inv_ker = self.inv_ker

        K2_ten = np.array([mat_kernel_func(X[i], Xtrain[j]) for i in np.arange(len(X)) for j in np.arange(Ntrain)])
        K2_ten.resize((len(X), Ntrain, D, D))
        '''
        localdet = (np.linalg.det(K2_ten[0,0]))**(1.0/3.0)
        print(K2_ten[0,0,:,:])
        print('Old determinant is:', np.linalg.det(K2_ten[0, 0]))
        print(K2_ten[0,0,:,:]/localdet)
        print('New determinant is', np.linalg.det((K2_ten[0,0]/localdet)))
        print('New trace is:', np.trace((K2_ten[0,0,:,:]/localdet)))
        print('alpha vector is:', self.alpha_ten)						### USED FOR DEBUGGING
		'''
        if self.m_theta0[0] is None:
            pred = np.einsum(' nNdp, Np -> nd', K2_ten, self.alpha_ten)
        else:
            m_theta = self.m_theta0[0]
            mvec_pred = np.array([repulsion(X[n], m_theta) for n in np.arange(len(X))])
            pred = mvec_pred + np.einsum(' nNdp, Np -> nd', K2_ten, self.alpha_ten)

        if calc_error == True:
            var = np.zeros((len(X), 3))

            for i in np.arange(len(X)):
                kv = K2_ten[i]
                var[i] = np.diag(mat_kernel_func(X[i], X[i]))
                var[i] += nugget
                var[i] = np.diag(np.einsum('Uab, UbDc, Ddc -> ad', kv, self.inv_ten, kv))

            return pred, var
        else:

            return pred

    def predict_energy(self, X):
        """
        This function evaluates the Gaussian Process model at X.
    
        Returns
        -------
        y : array_like
        """

        Xtrain = self.Xtrain
        Ntrain = self.Ntrain
        fvecs = self.fvecs
        self.X = X
        gamma = self.gamma
        calc_error = self.calc_error
        K2_ten = np.zeros((len(X), Ntrain, D))
        ker = self.ker
        theta0 = self.theta0
        sig = self.sig
        nugget = self.nugget
        mat_kernel_func = self.mat_kernel_func
        inv_ker = self.inv_ker
        if True:
            K2_ten = np.array(
                [energy_square(X[i], Xtrain[j], sig) for i in np.arange(len(X)) for j in np.arange(Ntrain)])

        K2_ten.resize((len(X), Ntrain, D))

        if self.m_theta0[0] is None:
            pred = np.einsum(' nNp, Np -> n', K2_ten, self.alpha_ten)
        else:
            m_theta = self.m_theta0[0]
            mvec_pred = np.array([repulsion_energy(X[n], m_theta) for n in np.arange(len(X))])
            pred = mvec_pred + np.einsum(' nNp, Np -> n', K2_ten, self.alpha_ten)

        return pred

    def update(self, new_x=None, new_y=None, index=0):

        sig = self.sig
        theta0 = self.theta0
        x_old = self.Xtrain
        Ntrain = self.Ntrain
        mat_kernel_func = self.mat_kernel_func
        y_old = self.ytrain
        eval_grad = self.eval_grad
        K_new = self.K
        y_ten = np.reshape(y_old, (Ntrain, D))
        y_ten[index, :] = new_y

        K_new[index * D:index * D + D, index * D:index * D + D] = mat_kernel_func(new_x, new_x)
        for i in np.arange(Ntrain):
            if (i != index):
                K_new[index * D:index * D + D, i * D:i * D + D] = mat_kernel_func(new_x, x_old[i])
                K_new[i * D:i * D + D, index * D:index * D + D] = K_new[index * D:index * D + D, i * D:i * D + D]

        # invert covariance matrix
        try:
            inv = LA.pinv2(K_new)
        except LA.LinAlgError as err:
            print("pinv2 failed: %s. Switching to pinvh" % err)
            try:
                inv = LA.pinvh(K_new)
            except LA.LinAlgError as err:
                print("pinvh failed: %s. Switching to pinv2" % err)
                inv = None

        # alpha is the vector of regression coefficients of GaussianProcess
        inv_ten = np.reshape(inv, (Ntrain, D, Ntrain, D))
        self.inv_ten = inv_ten
        self.K = K_new
        self.alpha_ten = np.einsum('ndND, ND -> nd ', inv_ten, y_ten)

    def save(self, name):
        K = self.K
        alpha_ten = self.alpha_ten
        Xtrain = self.Xtrain
        Ntrain = self.Ntrain

        output = []
        output.append(K)
        output.append(alpha_ten)
        output.append(Xtrain)
        output.append(Ntrain)
        np.save('%s' % name, output)
        print('Saved Gaussian process with name:', name)

    def load(self, in_data):
        a = np.load(in_data)
        self.K = np.asarray(a[0])
        self.alpha_ten = np.asarray(a[1])
        self.Xtrain = np.asarray(a[2])
        self.Ntrain = a[3]

        print('Loaded GP from file')
