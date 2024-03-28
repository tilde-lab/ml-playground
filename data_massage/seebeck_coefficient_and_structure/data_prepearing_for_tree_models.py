from mpds_client import MPDSDataRetrieval
from turicreate import SFrame
from descriptors.utils import get_APF, get_Wiener
import numpy as np
from ase import Atoms
from ase.data import chemical_symbols
import pandas as pd
from data_massage.mendeleev_table import periodic_numbers

def data_prepearing_from_file(file_path):
    """
    Prepare data to next format: "phase_id", "APF", "Wiener", "I", "C".
    Input file consist of next columns: "phase_id", "Formula", "Seebeck coefficient", "entry",
    "cell_abc", "sg_n", "basis_noneq", "els_noneq", "I", "C" (see ml-playground/data_massage/props.json).
    Return SFrame.
    """

    data = pd.read_excel(file_path)

    seebeck = pd.DataFrame(data, columns=["phase_id", "Seebeck coefficient"])
    seebeck["phase_id"] = data["phase_id"]
    seebeck["Seebeck coefficient"] = data["Seebeck coefficient"]

    data.drop(data.columns[2], axis=1, inplace=True)
    data[['Formula', 'entry']] = data[['entry', 'Formula']]

    # Converting structure dataframe to list for easier descriptor calculation
    structure_list = data.values.tolist()

    # Calculating decriptors
    descriptors = []
    new_list = []
    for value in structure_list:
        new_list.append(
            [value[0], value[1], value[2], eval(value[3]),
             value[4], eval(value[5]), eval(value[6]),
             value[7], value[8]]
        )

    for item in new_list:
        if item[4] != 1:
            crystal = MPDSDataRetrieval.compile_crystal(item[:-2], "ase")
            if not crystal:
                continue
            descriptors.append((item[0], get_APF(crystal), get_Wiener(crystal), item[7], item[8]))
        elif item[4] == 1:
            crystal = Atoms(symbols=item[6], positions=item[5], cell=item[3])
            if not crystal:
                continue
            descriptors.append((item[0], get_APF(crystal), get_Wiener(crystal), item[7], item[8]))

    descriptors = pd.DataFrame(descriptors, columns=["phase_id", "APF", "Wiener", "I", "C"])

    total = seebeck.merge(descriptors, on="phase_id")

    total = SFrame(data=total)
    return total

def data_prepearing_with_request():
    # Display options settings
    np.set_printoptions(suppress=True)
    pd.set_option("display.max_columns", None)

    # Data Retrieval
    client = MPDSDataRetrieval()

    # Physical Property Retrieval
    seebeck = client.get_data({"classes": "binary", "props": "seebeck coefficient"})
    seebeck = pd.DataFrame(
        seebeck, columns=["Phase", "Formula", "SG", "Entry", "Property", "Units", "Value"]
    )

    # Cleaning of Seebeck Dataframe
    seebeck = seebeck[np.isfinite(seebeck["Phase"])]
    seebeck = seebeck[seebeck["Units"] == "muV K-1"]  # cleaning

    # Making a list of all phase_ids
    phases = set(seebeck["Phase"].tolist())

    # Retrieving structural properties putting restriction on Phase
    structure = client.get_data(
        {
            "classes": "binary",
            "props": "structural properties",  # see https://mpds.io/#hierarchy
        },
        phases=phases,
        fields={
            "S": [
                "phase_id",
                "entry",
                "chemical_formula",
                "cell_abc",
                "sg_n",
                "basis_noneq",
                "els_noneq",
            ]
        },
    )

    structure = pd.DataFrame(
        structure,
        columns=["Phase", "Entry", "Formula", "cell_abc", "SG", "Basis_noneq", "els_noneq"],
    )

    # Data Cleansing
    structure = structure.dropna()
    seebeck = seebeck.dropna()
    structure = structure[structure.all(1)]
    seebeck = seebeck[seebeck["Phase"].isin(structure["Phase"])]

    # Selecting rows with unique phase values(picking first and removing the duplicate ones)
    structure = structure.sort_values("Phase", ascending=True)
    structure = structure.drop_duplicates(subset="Phase", keep="first")
    seebeck = seebeck.sort_values("Phase", ascending=True)
    seebeck = seebeck.drop_duplicates(subset="Phase", keep="first")

    # Converting structure dataframe to list for easier descriptor calculation
    structure_list = structure.values.tolist()

    # Calculating decriptors
    descriptors = []

    for item in structure_list:
        crystal = MPDSDataRetrieval.compile_crystal(item, "ase")
        if not crystal:
            continue
        descriptors.append((item[0], get_APF(crystal), get_Wiener(crystal)))

    descriptors = pd.DataFrame(descriptors, columns=["Phase", "APF", "Wiener"])

    total = structure.merge(seebeck, on="Phase").merge(descriptors, on="Phase")

    # Converting pandas dataframe to SFrame
    total = SFrame(data=total)

    return total

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

def to_cat_vectors_struct(path_to_save, file_path):
    data_csv = pd.read_csv(file_path)
    d_list = data_csv.values.tolist()

    objs = []
    seebeck = []

    for item in d_list:
        crystal = Atoms(symbols=eval(item[6]), positions=eval(item[5]), cell=eval(item[3]))
        vectors = get_bv_descriptor(crystal)
        if len(vectors[0]) < 32:
            continue
        elif len(vectors[0]) == 32:
            objs.append(vectors)
            seebeck.append(item[2])
        else:
            objs.append(vectors[:, :32])
            seebeck.append(item[2])


    dfrm = pd.DataFrame([i.tolist() for i in objs], columns=['atom', 'distance'])
    dfrm.to_csv(path_to_save+'rep_vectors_str_200.csv', index=False)

    dfrm_s = pd.DataFrame(seebeck, columns=['Seebeck coefficient'])
    dfrm_s.to_csv(path_to_save+'seebeck_200.csv', index=False)

path = '/root/projects/ml-playground/data_massage/seebeck_coefficient_and_structure/data/26_3/rep_ordered_str_200.csv'
path_to_save = '/root/projects/ml-playground/data_massage/seebeck_coefficient_and_structure/data/26_3/'

to_cat_vectors_struct(file_path=path, path_to_save=path_to_save)
print()