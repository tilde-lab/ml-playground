import ase.build
from ase.geometry.dimensionality import analyze_dimensionality
import numpy as np
import ase.build
from ase import Atoms
from ase.geometry.dimensionality import isolate_components

# Create structure.
atoms = ase.build.mx2(formula='MoS2', kind='2H', a=3.18, thickness=3.19)
atoms.cell[2, 2] = 7.0
atoms.set_pbc((1, 1, 1))
atoms *= 3

# Structure dimension analysis using RDA method.
intervals = analyze_dimensionality(atoms, method='RDA')
m = intervals[0]

# Score is a numerical indicator of how well does this component describe the structure.
print(sum([e.score for e in intervals]))
print(m.dimtype, m.h, m.score, m.a, m.b)


# Build two slabs of different types of MoS2.
rep = [4, 4, 1]
a = ase.build.mx2(formula='MoS2', kind='2H', a=3.18, thickness=3.19) * rep
b = ase.build.mx2(formula='MoS2', kind='1T', a=3.18, thickness=3.19) * rep
# Here, the positions and atomic numbers for the two created structures are concatenated.
# The second layer b goes along the Z axis by 7 units.
positions = np.concatenate([a.get_positions(), b.get_positions() + [0, 0, 7]])
numbers = np.concatenate([a.numbers, b.numbers])
cell = a.cell
# United structure.
atoms = Atoms(numbers=numbers, positions=positions, cell=cell, pbc=[1, 1, 1])
atoms.cell[2, 2] = 14.0

# isolate each component in the whole material
result = isolate_components(atoms)
print("counts:", [(k, len(v)) for k, v in sorted(result.items())])

for dim, components in result.items():
    for atoms in components:
        print(dim)
