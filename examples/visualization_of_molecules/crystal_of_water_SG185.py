from ase.spacegroup import crystal
from ase.io import write

a, b, c = 7.820, 7.820, 7.360
alpha, beta, gamma = 90, 90, 120

spacegroup = 185

basis = [(0.336017, 0.336017, 0.696031),
         (0.460401, 0.460401, 0.511393),
         (0.334231, 0.334231, 0.555157)]

h2o_crystal = crystal(symbols=['H', 'H', 'O'],
                      basis=basis,
                      spacegroup=spacegroup,
                      cellpar=[a, b, c, alpha, beta, gamma],
                      pbc=True)

h2o_crystal *= (1, 1, 1)

cif_file_path = f"crystal_water_mol_test.cif"
write(cif_file_path, h2o_crystal)
