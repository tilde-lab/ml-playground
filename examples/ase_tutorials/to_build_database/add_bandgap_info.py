from ase.db import connect
from gpaw import GPAW, PW
from ase.dft.bandgap import bandgap

db = connect('database.db')

# For all data which marked as relaxed:
for row in db.select(relaxed=True):
    atoms = row.toatoms()
    # Create calculator.
    calc = GPAW(mode=PW(400),
                kpts=(4, 4, 4),
                txt=f'{row.formula}-gpaw.txt', xc='LDA')

    # Store the bandgap information in db.
    atoms.calc = calc
    atoms.get_potential_energy()
    bg, cbm, vbm = bandgap(calc=atoms.calc)

    # Save data in db.
    db.update(row.id, bandgap=bg)