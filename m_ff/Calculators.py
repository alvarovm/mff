import numpy as np
from ase.calculators.calculator import Calculator, all_changes
from asap3 import FullNeighborList


# from ase.data import chemical_symbols, atomic_numbers
# from ase.units import Bohr
# from ase.neighborlist import NeighborList

class SingleSpecies(Exception):
    pass


class RemappedTwoBodySingleSpecies(Calculator):
    """A remapped 2-body calculator for ase
    """

    # 'Properties calculator can handle (energy, forces, ...)'
    implemented_properties = ['energy', 'forces']

    # 'Default parameters'
    default_parameters = {}

    def __init__(self, restart=None, ignore_bad_restart_file=False, label='abinit', atoms=None, **kwargs):
        super().__init__(self, restart, ignore_bad_restart_file, label, atoms, **kwargs)

    def calculate(self, atoms=None, properties=['energy'], system_changes=all_changes):
        """Do the calculation.

        properties: list of str
            List of what needs to be calculated.  Can be any combination
            of 'energy', 'forces', 'stress', 'dipole', 'charges', 'magmom'
            and 'magmoms'.
        system_changes: list of str
            List of what has changed since last calculation.  Can be
            any combination of these six: 'positions', 'numbers', 'cell',
            'pbc', 'initial_charges' and 'initial_magmoms'.

        Subclasses need to implement this, but can ignore properties
        and system_changes if they want.  Calculated properties should
        be inserted into results dictionary like shown in this dummy
        example::

            self.results = {'energy': 0.0,
                            'forces': np.zeros((len(atoms), 3)),
                            'stress': np.zeros(6),
                            'dipole': np.zeros(3),
                            'charges': np.zeros(len(atoms)),
                            'magmom': 0.0,
                            'magmoms': np.zeros(len(atoms))}

        The subclass implementation should first call this
        implementation to set the atoms attribute.
        """

        positions = self.atoms.positions
        numbers = self.atoms.numbers
        cell = self.atoms.cell

        if atoms is not None:
            self.atoms = atoms.copy()

        self.results['forces'] = None


if __name__ == '__main__':
    from ase import Atoms

    parameters = {'cut_off': pair_style, 'pair_coeff': pair_coeff}

    calc = RemappedTwoBodySingleSpecies(parameters=parameters, files=files)

    a0 = 3.93
    b0 = a0 / 2.0
    bulk = Atoms(
        ['C'] * 4,
        positions=[(0, 0, 0), (b0, b0, 0), (b0, 0, b0), (0, b0, b0)],
        cell=[a0] * 3, pbc=True)

    # test get_forces
    print('forces for a = {0}'.format(a0))
    print(calc.get_forces(bulk))
    # single points for various lattice constants
    bulk.set_calculator(calc)
    for n in range(-5, 5, 1):
        a = a0 * (1 + n / 100.0)
        bulk.set_cell([a] * 3)
        print('a : {0} , total energy : {1}'.format(
            a, bulk.get_potential_energy()))

    calc.clean()
