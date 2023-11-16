# This code is a study of the dependence of the energy of the crystal structure of atoms on the size of its cell.
# Changing the cell size and optimizing lattice parameters makes it possible to
# find parameters at which the energy of the structure is minimal.
# This is important for calculations, since the minimum energy corresponds to stable structure.

import numpy as np

from ase import Atoms
from ase.calculators.emt import EMT
from ase.io.trajectory import Trajectory

from ase.eos import EquationOfState
from ase.io import read
from ase.units import kJ

atoms_name = 'Ag'
a = 4.0  # approximate lattice constant
b = a / 2
cell_is = [(0, b, b), (b, 0, b), (b, b, 0)]

ag = Atoms(atoms_name,
           cell=cell_is,
           pbc=1,
           calculator=EMT())  # use EMT potential

cell = ag.get_cell()
traj = Trajectory('molecule.traj', 'w')

# Changing size of cell in distance from 0.95 to 1.05.
for x in np.linspace(0.95, 1.05, 5):
    # To install new size:
    ag.set_cell(cell * x, scale_atoms=True)
    # To calculate potential energy:
    ag.get_potential_energy()
    traj.write(ag)

configs = read('molecule.traj@0:5')  # read 5 configurations
# Extract volumes and energies:
volumes = [ag.get_volume() for ag in configs]
energies = [ag.get_potential_energy() for ag in configs]

# The equation of state is fitted to the data.
eos = EquationOfState(volumes, energies)
# Volume at equilibrium (v0), minimum energy (e0), and Birger's modulus of elasticity (B).
v0, e0, B = eos.fit()
print(B / kJ * 1.0e24, 'GPa')
eos.plot('Molec-eos.png')