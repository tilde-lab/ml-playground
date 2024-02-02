from ase.spacegroup import crystal
from ase.io import write
from ase.build import molecule

# just molecule
atoms = molecule('H2O')

# crystal
h2o_crystal = crystal(['H', 'H', 'O'],
                      basis=[(0.336017, 0.336017, 0.696031),
                             (0.460401, 0.460401, 0.511393),
                             (0.334231, 0.334231, 0.555157)],
                      spacegroup=1,
                      cellpar=[7.50, 7.50, 7.06, 90, 90, 107.4],
                      pbc=True)

# check angle from H2O
angle_2 = h2o_crystal.get_angle(1, 2, 0)
print('Angle between O-H in crystal:', angle_2)

h2o_crystal *= (6, 6, 6)

cif_file_path = f"crystal_water_mol_test.cif"
write(cif_file_path, h2o_crystal)
