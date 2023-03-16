from mpds_client import MPDSDataRetrieval
import numpy as np
import pandas as pd
from ase.data import chemical_symbols, covalent_radii
import turicreate as tc
from turicreate import SFrame

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

# Defining descriptors
def get_APF(ase_obj):
    volume = 0.0
    for atom in ase_obj:
        volume += (
            4 / 3 * np.pi * covalent_radii[chemical_symbols.index(atom.symbol)] ** 3
        )
    return volume / abs(np.linalg.det(ase_obj.cell))


def get_Wiener(ase_obj):
    return np.sum(ase_obj.get_all_distances()) * 0.5


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

# Any of the below regression models can be used for predicting Seebeck Coefficient values of binary compounds.

# LINEAR REGRESSION MODEL
model_linear = tc.linear_regression.create(
    total, target="Value", features=["APF", "Wiener"]
)
coefficients_linear = model_linear.coefficients
predictions_linear = model_linear.predict(total)
results_linear = model_linear.evaluate(total)
model_linear.summary()

# DECISION TREE MODEL
train_d, test_d = total.random_split(0.85)
model_decision = tc.decision_tree_regression.create(
    train_d, target="Value", features=["APF", "Wiener"]
)
predictions_decision = model_decision.predict(test_d)
results_decision = model_decision.evaluate(test_d)
featureimp_decision = model_decision.get_feature_importance()
model_decision.summary()

# BOOSTED TREES MODEL
train_b, test_b = total.random_split(0.85)
model_boosted = tc.boosted_trees_regression.create(
    train_b, target="Value", features=["APF", "Wiener"]
)
predictions_boosted = model_boosted.predict(test_b)
results_boosted = model_boosted.evaluate(test_b)
featureboosted = model_boosted.get_feature_importance()
model_boosted.summary()

# RANDOM FOREST MODEL
train_r, test_r = total.random_split(0.85)
model_random = tc.random_forest_regression.create(
    train_r, target="Value", features=["APF", "Wiener"]
)
predictions_random = model_random.predict(test_r)
results_random = model_random.evaluate(test_r)
featureimp_random = model_random.get_feature_importance()
model_random.summary()
