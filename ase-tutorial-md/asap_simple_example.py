# First a documentation string to describe the example
"""An example script demonstrating an NVT MD simulation with Asap.

This example runs a molecular dynamics simulation on a small copper
cluster in the NVT ensemble, i.e. with constant particle number,
constant volume and constant temperature.  It uses Langevin dynamics
to get the constant temperature.
"""

# Import ASAP and relatives
from ase import *
from asap3 import *
from ase.md.langevin import Langevin
from ase.lattice.cubic import FaceCenteredCubic
from ase.io.trajectory import Trajectory
from ase.units import kB
# Import other essential modules
from numpy import *

# Set up an fcc crystal of copper, 1372 atoms.
atoms = FaceCenteredCubic(size=(7, 7, 7), symbol="Cu", pbc=False)

# Now the standard EMT Calculator is attached
atoms.set_calculator(EMT())

# Make an object doing Langevin dynamics at a temperature of 800 K
dyn = Langevin(atoms, timestep=5 * units.fs, temperature=800 * units.kB, friction=0.005)

# Make a trajectory
traj = Trajectory('MD_Cluster.traj', "w", atoms)
dyn.attach(traj, interval=500)  # Automatically writes the initial configuration

step = 0


# Print the energies
def printenergy(a):
    global step

    n = len(a)
    ekin = a.get_kinetic_energy() / n
    epot = a.get_potential_energy() / n
    print("%4d: E_kin = %-9.5f  E_pot = %-9.5f  E_tot = %-9.5f  T = %.1f K" %
          (step, ekin, epot, ekin + epot, 2.0 / 3.0 * ekin / kB))

    step += 1


printenergy(atoms)

# Now do the dynamics, doing 5000 timesteps, writing energies every 50 steps
dyn.attach(printenergy, 50, atoms)
dyn.run(5000)

# Now increase the temperature to 2000 K and continue
dyn.set_temperature(2000 * units.kB)
dyn.run(5000)
