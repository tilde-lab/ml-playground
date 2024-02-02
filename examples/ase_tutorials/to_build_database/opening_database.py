from ase.db import connect

db = connect('database.db')

# Shows all rows in db (we omitted parameters db.select() for that).
for row in db.select():
    atoms = row.toatoms()
    print(atoms)

print('\n')

# Shows bandgap for structure.
for row in db.select(relaxed=True):
    formula = row.formula
    bandgap_value = row.bandgap
    print(formula, bandgap_value)

# # Shows documentation for 'select'.
# from ase.db import connect
# db = connect('database.db')
# help(db.select)
