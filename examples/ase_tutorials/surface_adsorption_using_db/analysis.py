# Script calculates the adsorption energy and height for each adsorbed atom structure
# in the 'ads.db' and updates the corresponding data with this info.

from ase.db import connect

refs = connect('refs.db')
db = connect('ads.db')

for row in db.select():
    # Calculates energy of adsorption.
    ea = (row.energy -
          refs.get(formula=row.ads).energy -
          refs.get(layers=row.layers, surf=row.surf).energy)
    # Calculates height of an adsorbed atom using coordinates of its position.
    h = row.positions[-1, 2] - row.positions[-2, 2]
    db.update(row.id, height=h, ea=ea)