# Equation of state is a mathematical relationship between volume and energy.
# Goal is to understand which cell volumes (and therefore which structures) are the most stable from an energetic.

# Use '$ ase db bulk.db' to see data in db.

from ase.build import bulk
from ase.calculators.emt import EMT
from ase.db import connect
from ase.eos import calculate_eos

db = connect('bulk.db')
fcc = ['Al', 'Ni', 'Cu', 'Pd', 'Ag', 'Pt', 'Au']

for symb in fcc:
    # Create crystal structure.
    atoms = bulk(symb, 'fcc')
    atoms.calc = EMT()
    eos = calculate_eos(atoms)
    # Using 'fit', parameters (volume, energy and compression modulus) are selected.
    v, e, B = eos.fit()  # find minimum
    # Do one more calculation at the minimu and write to database:
    atoms.cell *= (v / atoms.get_volume())**(1 / 3)
    atoms.get_potential_energy()
    # Save information about equation of state.
    db.write(atoms, bm=B)