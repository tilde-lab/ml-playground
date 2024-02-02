from optimade.client import OptimadeClient
from ase.io import read
from ase.visualize import view
import os
import json
from ase.io import write
from ase import Atom, Atoms

def optimade_to_ase(data, cnt):
    if 18 > len(data['species_at_sites']) > 10 :
        return False
    cnt += 1
    symbols = data['species_at_sites']
    positions = data['cartesian_site_positions']
    pbc = [False, False, False]
    atoms = Atoms(symbols=symbols, positions=positions, pbc=pbc)

    return atoms

client = OptimadeClient()
query = 'chemical_formula_reduced="H2O"'

folder_path = "/"
cnt = 0
cif_file_path = f"test_water_mol{cnt}.cif"
file_path = os.path.join(folder_path, cif_file_path)
output_file = "result.json"

results = client.get(filter=query)
structures = results['structures'][query]

# with open(output_file, "w") as f:
#     json.dump(structures, f)

with open(output_file, "r") as f:
    structures = json.load(f)

for structure in structures:
    if ('data' in structures[structure])!= []:
        for el in structures[structure]['data']:
            if 'attributes' in el:
                attributes = el['attributes']
                if ('cartesian_site_positions' in attributes):
                    positions = attributes['cartesian_site_positions']
                    ase_obj = optimade_to_ase(el['attributes'], cnt)
                    if ase_obj:
                        write(cif_file_path, ase_obj)
                        print(f"Molecular formula: {attributes['chemical_formula_reduced']}")
                        print(f"Atomic Positions: {positions}")
                        view(ase_obj)





