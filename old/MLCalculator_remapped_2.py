import numpy as np
from pathos.multiprocessing import ProcessingPool


class MLCalculator_local:

    def __init__(self, ff, ef, nl, gp=[]):
        self.nl = nl
        self.ff = ff
        self.ef = ef
        self.gp = gp

    def get_potential_energy(self, atoms):
        n = len(atoms)
        nl = self.nl
        nl.update(atoms)
        cell = atoms.get_cell()
        confs = []
        ef = self.ef

        for a in np.arange(n):
            indices, offsets = nl.get_neighbors(a)
            offsets = np.dot(offsets, cell)
            conf = np.zeros((len(indices), 3))
            for i, (a2, offset) in enumerate(zip(indices, offsets)):
                d = atoms.positions[a2] + offset - atoms.positions[a]
                conf[i] = d
            confs.append(conf)

        confs = np.array(confs)

        ### MULTIPROCESSING ### 

        if False:  ### Using the remapped energy ###
            nods = 2
            pool = ProcessingPool(nodes=nods)
            splitind = np.zeros(nods + 1)
            factor = (n + (nods - 1)) / nods
            splitind[1:-1] = [(i + 1) * factor for i in np.arange(nods - 1)]
            splitind[-1] = n
            splitind = splitind.astype(int)
            clist = [confs[splitind[i]:splitind[i + 1]] for i in np.arange(nods)]
            result_energy = np.array(pool.map(ef, clist))
            result_energy = np.concatenate(result_energy)

        else:
            result_energy = ef(confs)

        energies = np.reshape(result_energy, n)
        energy = -np.sum(energies)
        return energy

    def get_forces(self, atoms):

        ff = self.ff

        n = len(atoms)
        nl = self.nl
        nl.update(atoms)
        cell = atoms.get_cell()
        confs = []

        for a in np.arange(n):
            indices, offsets = nl.get_neighbors(a)
            offsets = np.dot(offsets, cell)
            conf = np.zeros((len(indices), 3))
            for i, (a2, offset) in enumerate(zip(indices, offsets)):
                d = atoms.positions[a2] + offset - atoms.positions[a]
                conf[i] = d
            confs.append(conf)

        confs = np.array(confs)

        ### MULTIPROCESSING ### 

        if False:  ### Using the remapped energy ###
            nods = 2
            pool = ProcessingPool(nodes=nods)
            splitind = np.zeros(nods + 1)
            factor = (n + (nods - 1)) / nods
            splitind[1:-1] = [(i + 1) * factor for i in np.arange(nods - 1)]
            splitind[-1] = n
            splitind = splitind.astype(int)
            clist = [confs[splitind[i]:splitind[i + 1]] for i in np.arange(nods)]
            result_force = np.array(pool.map(ff, clist))
            result_force = np.concatenate(result_force)

        else:
            result_force = ff(confs)

        forces = np.reshape(result_force, (n, 3))
        return forces
