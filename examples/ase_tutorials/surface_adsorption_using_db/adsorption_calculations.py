from ase.build import add_adsorbate, fcc111
from ase.calculators.emt import EMT
from ase.constraints import FixAtoms
from ase.db import connect
from ase.optimize import BFGS

db1 = connect('bulk.db')
db2 = connect('ads.db')


def run(symb, a, n, ads):
    # Create structure.
    atoms = fcc111(symb, (1, 1, n), a=a)
    # Adsorbate “sticks” to surface of structure.
    add_adsorbate(atoms, ads, height=1.0, position='fcc')

    # Constrain all atoms except the adsorbate:
    fixed = list(range(len(atoms) - 1))
    atoms.constraints = [FixAtoms(indices=fixed)]

    atoms.calc = EMT()
    # Try to minimize energy.
    opt = BFGS(atoms, logfile=None)
    opt.run(fmax=0.01)

    return atoms

# For all data in db:
for row in db1.select():
    a = row.cell[0, 1] * 2
    symb = row.symbols[0]
    # Value 'n' represents number of surface layers to which the adsorbate will be added.
    for n in [1, 2, 3]:
        # For each atom in 'C', 'N', 'O' (adsorbates).
        for ads in 'CNO':
            id = db2.reserve(layers=n, surf=symb, ads=ads)
            if id is not None:
                # Adsorbate addition, constraint application, and structure optimization.
                atoms = run(symb, a, n, ads)
                db2.write(atoms, layers=n, surf=symb, ads=ads)
                del db2[id]

