from ase import Atoms
from ase.calculators.emt import EMT

atoms_name = 'H'
molecules_formula = '2H'
# length between atoms (varies for a specific case: 0.74 (2H), 1.1 (2N))
d = 0.74
positions = [(0., 0., 0.), (0., 0., d)]

# create atom
atom = Atoms(atoms_name)
atom.calc = EMT()
e_atom = atom.get_potential_energy()

# create molecule
molecule = Atoms(molecules_formula, positions=positions)
molecule.calc = EMT()
e_molecule = molecule.get_potential_energy()

# if energy is positive, this means that the molecule is more stable than its atoms in an isolated state
e_atomization = e_molecule - 2 * e_atom

print('Atom energy: %5.2f eV' % e_atom)
print('Molecule energy: %5.2f eV' % e_molecule)
print('Atomization energy: %5.2f eV' % -e_atomization)