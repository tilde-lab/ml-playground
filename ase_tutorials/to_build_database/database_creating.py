# After running that code, you can check db by command '$ ase db database.db'.
# For make a view of all structure in db we can use '$ ase gui database.db'.
# By the way, to see help use '$ ase db database.db -h'

from ase.build import bulk
from ase.db import connect

si = bulk('Si', crystalstructure='diamond', a=None)
ge = bulk('Ge', crystalstructure='diamond', a=None)
c = bulk('C', crystalstructure='diamond', a=None)

# Saving structure to db.
db = connect('database.db')
db.write(si, name='Si')
db.write(ge, name='Ge')
db.write(c, name='Mg')