import numpy as np
from ase.calculators.calculator import Calculator, all_changes
from asap3 import FullNeighborList


# from ase.data import chemical_symbols, atomic_numbers
# from ase.units import Bohr
# from ase.neighborlist import NeighborList

class SingleSpecies(Exception):
    pass


class MinimalCalculator:
    """ASE calculator.

    A calculator should store a copy of the atoms object used for the
    last calculation.  When one of the *get_potential_energy*,
    *get_forces*, or *get_stress* methods is called, the calculator
    should check if anything has changed since the last calculation
    and only do the calculation if it's really needed.  Two sets of
    atoms are considered identical if they have the same positions,
    atomic numbers, unit cell and periodic boundary conditions."""

    def get_potential_energy(self, atoms=None, force_consistent=False):
        """Return total energy.

        Both the energy extrapolated to zero Kelvin and the energy
        consistent with the forces (the free energy) can be
        returned."""
        return 0.0

    def get_forces(self, atoms):
        """Return the forces."""
        return np.zeros((len(atoms), 3))

    def get_stress(self, atoms):
        """Return the stress."""
        return np.zeros(6)

    def calculation_required(self, atoms, quantities):
        """Check if a calculation is required.

        Check if the quantities in the *quantities* list have already
        been calculated for the atomic configuration *atoms*.  The
        quantities can be one or more of: 'energy', 'forces', 'stress',
        'charges' and 'magmoms'.

        This method is used to check if a quantity is available without
        further calculations.  For this reason, calculators should
        react to unknown/unsupported quantities by returning True,
        indicating that the quantity is *not* available."""
        return False


class RemappedTwoBodySingleSpecies(Calculator):
    """A remapped 2-body calculator for ase
    """

    # 'Properties calculator can handle (energy, forces, ...)'
    implemented_properties = ['energy', 'forces']

    # 'Default parameters'
    default_parameters = {}

    def __init__(self, restart=None, ignore_bad_restart_file=False, label='abinit', atoms=None, **kwargs):
        super().__init__(self, restart, ignore_bad_restart_file, label, atoms, **kwargs)

        self.r_cut = None

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

    def conf_iterator(self, atoms):
        # https://wiki.fysik.dtu.dk/asap/Neighbor%20lists

        atomic_numbers = atoms.get_array('numbers', copy=False)
        nl = FullNeighborList(self.r_cut, atoms=atoms)

        for atom in atoms:
            inds, confs, ds = nl.get_neighbors(atom.index)

            yield atomic_numbers[inds], confs


if __name__ == '__main__':
    from ase import Atoms

    # Usual cutoff values:
    # Fe_vac: 4.5
    # BIP_300: 100 (practically inf)
    # HNi: 4.5
    # C_a: 3.2


    parameters = {
        'cutoff_radius': 3.2
    }

    calc = RemappedTwoBodySingleSpecies(parameters=parameters)

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
