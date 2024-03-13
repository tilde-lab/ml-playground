from mpds_client import MPDSDataRetrieval
import pandas as pd
from turicreate import SFrame
from descriptors.utils import get_APF, get_Wiener
from ase import Atoms
import numpy as np

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
