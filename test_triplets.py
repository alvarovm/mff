import numpy as np
from itertools import combinations, islice

from ase.io import iread, read
from asap3 import FullNeighborList


# from scipy.spatial import cKDTree

def find_triplets(atoms, nl):
    indices, distances, positions = [], [], dict()

    for i in range(len(atoms)):

        inds, pos, dists = nl.get_neighbors(i)
        assert len(inds) is len(np.unique(inds)), "There are repetitive indices!\n{}".format(inds)

        # ingnore already visited atoms
        inds, pos, dists = inds[inds > i], pos[inds > i, :], dists[inds > i]

        for local_ind, (j, pos_ij, dist_ij) in enumerate(zip(inds, pos, dists)):

            # Caching local displacement vectors
            positions[(i, j)], positions[(j, i)] = pos_ij, -pos_ij

            for k, dist_ik in islice(zip(inds, dists), local_ind + 1, None):

                try:
                    jk_ind = list(nl[j]).index(k)
                except ValueError:
                    continue  # no valid triplet

                _, j_pos, j_dists = nl.get_neighbors(j)

                indices.append([i, j, k])
                distances.append([dist_ij, dist_ik, j_dists[jk_ind]])

    return np.array(indices), np.sqrt(np.array(distances)), positions


def find_triplets2(atoms, nl):
    indices, distances, positions = [], [], dict()

    # caching
    arr = [nl.get_neighbors(i) for i in range(len(atoms))]

    for i, (inds, pos, dists) in enumerate(arr):
        # assert len(inds) is len(np.unique(inds)), "There are repetitive indices!\n{}".format(inds)

        # ingnore already visited nodes
        inds, pos, dists = inds[inds > i], pos[inds > i, :], dists[inds > i]

        for (j_ind, j), (k_ind, k) in combinations(enumerate(inds[inds > i]), 2):

            jk_ind, = np.where(arr[j][0] == k)

            if not jk_ind.size:
                continue  # no valid triplet

            indices.append([i, j, k])

            # Caching local position vectors
            positions[(i, j)], positions[(j, i)] = pos[j_ind], -pos[j_ind]
            positions[(i, k)], positions[(k, i)] = pos[k_ind], -pos[k_ind]
            positions[(j, k)], positions[(k, j)] = arr[j][1][jk_ind[0], :], -arr[j][1][jk_ind[0], :]

            distances.append([dists[j_ind], dists[k_ind], arr[j][2][jk_ind[0]]])

    return np.array(indices), np.sqrt(np.array(distances)), positions


if __name__ == '__main__':
    testfiles = {
        'BIP_300': 'test/data/BIP_300/movie.xyz',
        'C_a': 'test/data/C_a/data_C.xyz',
        'Fe_vac': 'test/data/Fe_vad/vaca_iron500.xyz',
        'HNi': 'test/data/HNI/h_ase500.xyz'
    }

    filename = 'test/data/Fe_vac/vaca_iron500.xyz'
    traj = read(filename, index=slice(None))

    r_cut = 3.5
    atoms = traj[-1]
    nl = FullNeighborList(r_cut, atoms=atoms)

    indices, distances, positions = find_triplets(atoms, nl)
    indices, distances, positions = find_triplets(atoms, nl)
    indices, distances, positions = find_triplets(atoms, nl)
    print(indices.shape, distances.shape, len(positions))

    indices2, distances2, positions2 = find_triplets2(atoms, nl)
    indices2, distances2, positions2 = find_triplets2(atoms, nl)
    indices2, distances2, positions2 = find_triplets2(atoms, nl)
    print(indices2.shape, distances2.shape, len(positions2))

    print(np.allclose(indices, indices2), np.allclose(distances, distances2))
