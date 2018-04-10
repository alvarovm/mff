# import theano
import numpy as np
import _pickle as cPickle

import os


class Kernel(object):
    pass


class TwoBodySingleSpecies(Kernel):
    pass


class TwoBodyTwoSpecies(Kernel):
    pass


class ThreeBodySingleSpecies(Kernel):
    pass


class ThreeBodyTwoSpecies(Kernel):
    pass


### Import Theano functions ###

theano_dir = os.path.dirname(os.path.abspath(__file__)) + '/theano_funcs/'

# two body
f = open(theano_dir + '2B_ms_ff.save', 'rb')
twobody_ff_T = cPickle.load(f, encoding='latin1')
f.close()

f = open(theano_dir + '2B_ms_ef.save', 'rb')
twobody_ef_T = cPickle.load(f, encoding='latin1')
f.close()

f = open(theano_dir + '2B_ms_ee.save', 'rb')
twobody_ee_T = cPickle.load(f, encoding='latin1')
f.close()

f = open(theano_dir + '2B_ms_for_ff.save', 'rb')
twobody_for_ff_T = cPickle.load(f, encoding='latin1')
f.close()

f = open(theano_dir + '2B_ms_for_ef.save', 'rb')
twobody_for_ef_T = cPickle.load(f, encoding='latin1')
f.close()

f = open(theano_dir + '2B_ms_for_ee.save', 'rb')
twobody_for_ee_T = cPickle.load(f, encoding='latin1')
f.close()

# three body
f = open(theano_dir + 'H3b_ms.save', 'rb')
threebody_ff_T = cPickle.load(f, encoding='latin1')
f.close()

f = open(theano_dir + 'G3b_ms.save', 'rb')
threebody_ef_T = cPickle.load(f, encoding='latin1')
f.close()

f = open(theano_dir + 'S3b_ms.save', 'rb')
threebody_ee_T = cPickle.load(f, encoding='latin1')
f.close()

f = open(theano_dir + '3B_ff_cut.save', 'rb')
threebody_ff_T_cut = cPickle.load(f, encoding='latin1')
f.close()

f = open(theano_dir + '3B_ef_cut.save', 'rb')
threebody_ef_T_cut = cPickle.load(f, encoding='latin1')
f.close()

f = open(theano_dir + '3B_ee_cut.save', 'rb')
threebody_ee_T_cut = cPickle.load(f, encoding='latin1')
f.close()


### Define wrappers around Theano functions ###

# two body
def twobody_ff(a, b, sig, theta, rc):
    ret = twobody_ff_T(np.zeros(3), np.zeros(3), a, b, sig, theta, rc)
    return ret


def twobody_ef(a, b, sig, theta, rc):
    ret = twobody_ef_T(np.zeros(3), np.zeros(3), a, b, sig, theta, rc)
    return ret


def twobody_ee(a, b, sig, theta, rc):
    ret = twobody_ee_T(np.zeros(3), np.zeros(3), a, b, sig, theta, rc)
    return ret


def twobody_for_ff(a_l, b_l, sig, theta, rc):
    ret = twobody_for_ff_T(np.zeros(3), np.zeros(3), a_l, b_l, sig, theta, rc)
    return ret


def twobody_for_ef(a_l, b_l, sig, theta, rc):
    ret = twobody_for_ef_T(np.zeros(3), np.zeros(3), a_l, b_l, sig, theta, rc)
    return ret


def twobody_for_ee(a_l, b_l, sig, theta, rc):
    ret = twobody_for_ee_T(np.zeros(3), np.zeros(3), a_l, b_l, sig, theta, rc)
    return ret


# three body

def threebody_ff(a, b, sig, theta, rc):
    ret = threebody_ff_T(np.zeros(3), np.zeros(3), a, b, sig)
    return ret


def threebody_ef(a, b, sig, theta, rc):
    ret = threebody_ef_T(np.zeros(3), np.zeros(3), a, b, sig)
    return ret


def threebody_ee(a, b, sig, theta, rc):
    ret = threebody_ee_T(np.zeros(3), np.zeros(3), a, b, sig)
    return ret


def threebody_ff_cut(a, b, sig, rc, gamma):
    ret = threebody_ff_T_cut(np.zeros(3), np.zeros(3), a, b, sig, gamma, rc)
    return ret


def threebody_ef_cut(a, b, sig, rc, gamma):
    ret = threebody_ef_T_cut(np.zeros(3), np.zeros(3), a, b, sig, gamma, rc)
    return ret


def threebody_ee_cut(a, b, sig, rc, gamma):
    ret = threebody_ee_T_cut(np.zeros(3), np.zeros(3), a, b, sig, gamma, rc)
    return ret


# Classes for 2 and 3 body kernels
class TwoBody(object):
    """Two body kernel.

    Parameters
    ----------
    theta[0]: lengthscale
    """

    def __init__(self, theta=[1.], bounds=[(1e-2, 1e2)]):
        self.theta = theta
        self.bounds = bounds

    def calc(self, X1, X2):

        K_trans = np.zeros((X1.shape[0] * 3, X2.shape[0] * 3))

        for i in np.arange(X1.shape[0]):
            for j in np.arange(X2.shape[0]):
                K_trans[3 * i:3 * i + 3, 3 * j:3 * j + 3] = twobody_ff(X1[i], X2[j], self.theta[0], self.theta[1],
                                                                       self.theta[2])

        return K_trans

    def calc_gram(self, X, eval_gradient=False):

        diag = np.zeros((X.shape[0] * 3, X.shape[0] * 3))
        off_diag = np.zeros((X.shape[0] * 3, X.shape[0] * 3))

        if eval_gradient:
            print('ERROR: GRADIENT NOT IMPLEMENTED YET')
            quit()

        else:
            for i in np.arange(X.shape[0]):
                diag[3 * i:3 * i + 3, 3 * i:3 * i + 3] = twobody_ff(X[i], X[i], self.theta[0], self.theta[1],
                                                                    self.theta[2])
                for j in np.arange(i):
                    off_diag[3 * i:3 * i + 3, 3 * j:3 * j + 3] = twobody_ff(X[i], X[j], self.theta[0], self.theta[1],
                                                                            self.theta[2])

            gram = diag + off_diag + off_diag.T

            return gram

    def calc_gram_future(self, X, eval_gradient=False):

        diag = np.zeros((X.shape[0] * 3, X.shape[0] * 3))
        off_diag = np.zeros((X.shape[0] * 3, X.shape[0] * 3))

        if eval_gradient:
            print('ERROR: GRADIENT NOT IMPLEMENTED YET')
            quit()

        else:
            gram = twobody_for_ff(X, X, self.theta[0], self.theta[1], self.theta[2])
            gram = np.reshape(gram.swapaxes(1, 2), (gram.shape[0] * gram.shape[2], gram.shape[1] * gram.shape[2]))
            return gram

    def calc_diag(self, X):

        diag = np.zeros((X.shape[0] * 3))

        for i in np.arange(X.shape[0]):
            diag[i * 3:(i + 1) * 3] = np.diag(twobody_ff(X[i], X[i], self.theta[0]))

        return diag

    def calc_ef(self, X1, X2):

        K_trans = np.zeros((X1.shape[0], X2.shape[0] * 3))

        for i in np.arange(X1.shape[0]):
            for j in np.arange(X2.shape[0]):
                K_trans[i, 3 * j:3 * j + 3] = twobody_ef(X1[i], X2[j], self.theta[0], self.theta[1], self.theta[2])

        return K_trans

    def calc_ef_future(self, X1, X2):

        K_trans = np.zeros((X1.shape[0], X2.shape[0] * 3))

        K_trans = twobody_for_ff(X1, X2, self.theta[0], self.theta[1], self.theta[2])
        K_trans = np.reshape(K_trans.swapaxes(1, 2),
                             (K_trans.shape[0] * K_trans.shape[2], K_trans.shape[1] * K_trans.shape[2]))
        return K_trans

        return K_trans

    def calc_diag_e(self, X):

        diag = np.zeros((X.shape[0]))

        for i in np.arange(X.shape[0]):
            diag[i] = twobody_ee(X[i], X[i], self.theta[0])

        return diag


class ThreeBody(object):
    """Three body kernel.

    Parameters
    ----------
    theta[0]: lengthscale
    theta[1]: hardness of cutoff function (to be implemented)
    """

    def __init__(self, theta=[None, None], bounds=[(1e-2, 1e2), (1e-2, 1e2)]):
        self.theta = theta
        self.bounds = bounds

    def calc(self, X1, X2):

        K_trans = np.zeros((X1.shape[0] * 3, X2.shape[0] * 3))

        for i in np.arange(X1.shape[0]):
            for j in np.arange(X2.shape[0]):
                K_trans[3 * i:3 * i + 3, 3 * j:3 * j + 3] = threebody_ff(X1[i], X2[j], self.theta[0], self.theta[1],
                                                                         self.theta[2])

        return K_trans

    def calc_gram(self, X, eval_gradient=False):

        diag = np.zeros((X.shape[0] * 3, X.shape[0] * 3))
        off_diag = np.zeros((X.shape[0] * 3, X.shape[0] * 3))

        if eval_gradient:
            print('ERROR: GRADIENT NOT IMPLEMENTED YET')
            quit()
        else:

            for i in np.arange(X.shape[0]):
                diag[3 * i:3 * i + 3, 3 * i:3 * i + 3] = threebody_ff(X[i], X[i], self.theta[0], self.theta[1],
                                                                      self.theta[2])
                for j in np.arange(i):
                    off_diag[3 * i:3 * i + 3, 3 * j:3 * j + 3] = threebody_ff(X[i], X[j], self.theta[0], self.theta[1],
                                                                              self.theta[2])

            gram = diag + off_diag + off_diag.T

            return gram

    def calc_diag(self, X):

        diag = np.zeros((X.shape[0] * 3))

        for i in np.arange(X.shape[0]):
            diag[i * 3:(i + 1) * 3] = np.diag(threebody_ff(X[i], X[i], self.theta[0], self.theta[1], self.theta[2]))

        return diag

    def calc_ef(self, X1, X2):

        K_trans = np.zeros((X1.shape[0], X2.shape[0] * 3))

        for i in np.arange(X1.shape[0]):
            for j in np.arange(X2.shape[0]):
                K_trans[i, 3 * j:3 * j + 3] = threebody_ef(X1[i], X2[j], self.theta[0], self.theta[1], self.theta[2])

        return K_trans

    def calc_diag_e(self, X):

        diag = np.zeros((X.shape[0]))

        for i in np.arange(X.shape[0]):
            diag[i] = threebody_ee(X[i], X[i], self.theta[0], self.theta[1], self.theta[2])

        return diag