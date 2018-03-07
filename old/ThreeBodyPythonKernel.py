import numpy as np
from scipy.spatial.distance import cdist


def myker(a, b, sig):
    # Kernel that considers valid triplets the ones which have all distances smaller than cut

    sigsq = sig * sig
    ri, rk = np.sqrt((a ** 2).sum(1)), np.sqrt((b ** 2).sum(1))

    # Distances
    vi = np.einsum('nd,n->nd', a, 1 / ri)  # Versors of a
    vk = np.einsum('nd,n->nd', b, 1 / rk)  # Versors of b
    outer = np.einsum('na, mb -> nmab', vi, vk)

    # Outer product of versors
    rij = np.array(cdist(a, a, 'euclidean'))
    # Matrix of distances within a, shape la*la
    rkl = np.array(cdist(b, b, 'euclidean'))
    # Matrix of distances within b, shape lb*lb

    distik = ri[:, None] - rk[None, :]  # Matrix of differences between ri and rk
    distijkl = rij[:, :, None, None] - rkl[None, None, :, :]

    # Matrix of differences between rij and rkl
    distijk = rij[:, :, None] - rk[None, None, :]  # Matrix of differences between rij and rk
    distikl = ri[:, None, None] - rkl[None, :, :]  # Matrix of differences between ri and rkl

    # SQ exp kernel between radial distances of A and B
    seik = np.exp(-np.square(distik) / (2 * sigsq))

    # SQ exp kernel between distances of atoms in A and B
    seijkl = np.exp(-np.square(distijkl) / (2 * sigsq))

    # SQ exp kernel between distances of atoms in A and radial distances of B
    seijk = np.exp(-np.square(distijk) / (2 * sigsq))

    # SQ exp kernel between radial distances of A and distances of atoms in B
    seikl = np.exp(-np.square(distikl) / (2 * sigsq))

    One = np.einsum('ik, ijkl, jl-> ijkl', seik, seijkl, seik)
    Two = np.einsum('ikl, ijl, jk-> ijkl', seikl, seijk, seik)
    Three = np.einsum('il, ijk, jkl-> ijkl', seik, seijk, seikl)

    F1 = -(np.einsum('ijkl, ik, ik->ijkl', One, distik, distik) +
           np.einsum('ijkl, ikl, jk->ijkl', Two, distikl, distik) +
           np.einsum('ijkl, il, ijk->ijkl', Three, distik, distijk)) / (sigsq * sigsq) + One / sigsq
    F2 = -(np.einsum('ijkl, jl, ik->ijkl', One, distik, distik) +
           np.einsum('ijkl, jk, jk->ijkl', Two, distik, distik) +
           np.einsum('ijkl, jkl, ijk->ijkl', Three, distikl, distijk)) / (sigsq * sigsq) + Two / sigsq
    F3 = -(np.einsum('ijkl, ik, jl->ijkl', One, distik, distik) +
           np.einsum('ijkl, ikl, ijl->ijkl', Two, distikl, distijk) +
           np.einsum('ijkl, il, il->ijkl', Three, distik, distik)) / (sigsq * sigsq) + Three / sigsq
    F4 = -(np.einsum('ijkl, jl, jl->ijkl', One, distik, distik) +
           np.einsum('ijkl, jk, ijl->ijkl', Two, distik, distijk) +
           np.einsum('ijkl, jkl, il->ijkl', Three, distikl, distik)) / (sigsq * sigsq) + One / sigsq

    EdAdB = np.einsum('ijkl, ikab->ijklab', F1, outer) + \
            np.einsum('ijkl, jkab->ijklab', F2, outer) + \
            np.einsum('ijkl, ilab->ijklab', F3, outer) + \
            np.einsum('ijkl, jlab->ijklab', F4, outer)

    # Double Derivative of the kernel on energy without cutoff

    simo = np.einsum('ijklab -> ab', EdAdB)

    return simo
