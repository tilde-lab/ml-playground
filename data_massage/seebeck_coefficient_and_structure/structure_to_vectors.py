import numpy as np
from ase import Atoms
from ase.data import chemical_symbols
import pandas as pd
from data_massage.mendeleev_table import periodic_numbers

def get_bv_descriptor(ase_obj, kappa=None, overreach=False):
    """
    From ASE object obtain a vectorized atomic structure
    populated to a certain fixed relatively big volume
    defined by kappa
    """
    if not kappa: kappa = 18
    if overreach: kappa *= 2

    norms = np.array([np.linalg.norm(vec) for vec in ase_obj.get_cell()])
    multiple = np.ceil(kappa / norms).astype(int)
    ase_obj = ase_obj.repeat(multiple)
    com = ase_obj.get_center_of_mass()
    ase_obj.translate(-com)
    del ase_obj[
        [atom.index for atom in ase_obj if np.sqrt(np.dot(atom.position, atom.position)) > kappa]
    ]

    ase_obj.center()
    ase_obj.set_pbc((False, False, False))
    sorted_seq = np.argsort(np.fromiter((np.sqrt(np.dot(x, x)) for x in ase_obj.positions), np.float))
    ase_obj = ase_obj[sorted_seq]

    elements, positions = [], []
    for atom in ase_obj:
        elements.append(periodic_numbers[chemical_symbols.index(atom.symbol)] - 1)
        positions.append(
            int(round(np.sqrt(atom.position[0] ** 2 + atom.position[1] ** 2 + atom.position[2] ** 2) * 10)))

    return np.array([elements, positions])

def to_cut_vectors_struct(path_to_save, file_path):
    data_csv = pd.read_csv(file_path)
    d_list = data_csv.values.tolist()

    objs = []
    seebeck = []

    for item in d_list:
        crystal = Atoms(symbols=eval(item[6]), positions=eval(item[5]), cell=eval(item[3]))
        vectors = get_bv_descriptor(crystal, kappa=40)
        if len(vectors[0]) < 100:
            continue
        elif len(vectors[0]) == 100:
            objs.append(vectors)
            seebeck.append(item[2])
        else:
            objs.append(vectors[:, :100])
            seebeck.append(item[2])


    dfrm = pd.DataFrame([i.tolist() for i in objs], columns=['atom', 'distance'])
    dfrm.to_csv(path_to_save+'vectors_str_200.csv', index=False)

    dfrm_s = pd.DataFrame(seebeck, columns=['Seebeck coefficient'])
    dfrm_s.to_csv(path_to_save+'rep_seebeck_200.csv', index=False)

path = \
    '/root/projects/ml-playground/data_massage/seebeck_coefficient_and_structure/data/26_3/median_rep_ordered_str_200.csv'
path_to_save = '/root/projects/ml-playground/data_massage/seebeck_coefficient_and_structure/data/01_04/'

to_cut_vectors_struct(file_path=path, path_to_save=path_to_save)
print()