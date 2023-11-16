from ase.db import connect
from ase.optimize import BFGS

# You need to install PAW (dataset) separately. Looks documentations here https://wiki.fysik.dtu.dk/gpaw/install.html
from gpaw import GPAW, PW
from ase.constraints import ExpCellFilter

db = connect('database.db')

# To loop all materials in the db.
for row in db.select():
    atoms = row.toatoms()

    # Create GPAW calculator.
    calc = GPAW(mode=PW(400),
                kpts=(4, 4, 4),
                txt=f'{row.formula}-gpaw.txt', xc='LDA')
    atoms.calc = calc
    atoms.get_stress()

    # Create to relax the cell parameters.
    filter = ExpCellFilter(atoms)

    # Optimize the structure.
    relaxing = BFGS(filter)
    relaxing.run(fmax=0.05)

    db.write(atoms=atoms, relaxed=True)
