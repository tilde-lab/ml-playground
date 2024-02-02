from mpds_client import MPDSDataRetrieval
import numpy as np
import pandas as pd
import turicreate as tc
from turicreate import SFrame
import pickle

from descriptors.utils import get_APF, get_Wiener

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

poly_descriptor = pickle.load(open("descriptor.p", "rb"))


row_file = open("rows.txt", "r")
row = row_file.read().rsplit("\n")
row = row[:-1]  # to remove empty item in the end of the list
row = np.array(row)
row_file.close()

dict = {}
for element in row:
    for entry in poly_descriptor:
        dict[element] = entry


ultimate_des = {}
for array in structure["els_noneq"]:
    values = np.empty([1, len(row)])
    for element in array:
        value = np.empty([1, len(row)])
        for key in dict:
            if element == key:
                value = dict[key]
                break
    values = np.add(values, value)
    for phase in structure["Phase"]:
        ultimate_des[phase] = values[0]

# Calculating decriptors
descriptors = []

for item in structure_list:
    crystal = MPDSDataRetrieval.compile_crystal(item, "ase")
    if not crystal:
        continue
    descriptors.append(
        (item[0], get_APF(crystal), get_Wiener(crystal), ultimate_des[item[0]])
    )

descriptors = pd.DataFrame(descriptors, columns=["Phase", "APF", "Wiener", "poly_info"])

total = structure.merge(seebeck, on="Phase").merge(descriptors, on="Phase")

total = structure.merge(seebeck, on="Phase").merge(descriptors, on="Phase")

# Converting pandas dataframe to SFrame
total = SFrame(data=total)
print(total)

# Any of the below regression models can be used for predicting Seebeck Coefficient values of binary compounds.

# LINEAR REGRESSION MODEL
model_linear1 = tc.linear_regression.create(
    total, target="Value", features=["APF", "Wiener"]
)
predictions_linear1 = model_linear1.predict(total)
results_linear1 = model_linear1.evaluate(total)

model_linear2 = tc.linear_regression.create(
    total, target="Value", features=["APF", "Wiener", "poly_info"]
)
predictions_linear2 = model_linear2.predict(total)
results_linear2 = model_linear2.evaluate(total)


# DECISION TREE MODEL
train_d, test_d = total.random_split(0.85)
model_decision1 = tc.decision_tree_regression.create(
    train_d, target="Value", features=["APF", "Wiener"]
)
predictions_decision1 = model_decision1.predict(test_d)
results_decision1 = model_decision1.evaluate(test_d)

model_decision2 = tc.decision_tree_regression.create(
    train_d, target="Value", features=["APF", "Wiener", "poly_info"]
)
predictions_decision2 = model_decision2.predict(test_d)
results_decision2 = model_decision2.evaluate(test_d)


# BOOSTED TREES MODEL
train_b, test_b = total.random_split(0.85)
model_boosted1 = tc.boosted_trees_regression.create(
    train_b, target="Value", features=["APF", "Wiener"]
)
predictions_boosted1 = model_boosted1.predict(test_b)
results_boosted1 = model_boosted1.evaluate(test_b)

model_boosted2 = tc.boosted_trees_regression.create(
    train_b, target="Value", features=["APF", "Wiener", "poly_info"]
)
predictions_boosted2 = model_boosted2.predict(test_b)
results_boosted2 = model_boosted2.evaluate(test_b)

# RANDOM FOREST MODEL
train_r, test_r = total.random_split(0.85)
model_random1 = tc.random_forest_regression.create(
    train_r, target="Value", features=["APF", "Wiener"]
)
predictions_random1 = model_random1.predict(test_r)
results_random1 = model_random1.evaluate(test_r)

model_random2 = tc.random_forest_regression.create(
    train_r, target="Value", features=["APF", "Wiener", "poly_info"]
)
predictions_random2 = model_random2.predict(test_r)
results_random2 = model_random2.evaluate(test_r)

# Results
print("linear_model1 result:", results_linear1)
print("linear_model2 result:", results_linear2)
print("decision_model1 result:", results_decision1)
print("decision_model2 result:", results_decision2)
print("boosted_model1 result:", results_boosted1)
print("boosted_model2 result:", results_boosted2)
print("Random_model1 result:", results_random1)
print("Random_model2 result:", results_random2)
