import pandas as pd

seebeck_coefficients = {
    'Se': 900, 'Te': 500, 'Si': 440, 'Ge': 330, 'Sb': 47, 'Fe': 19,
    'Mo': 10, 'Cd': 7.5, 'W': 7.5, 'Au': 6.5, 'Ag': 6.5, 'Cu': 6.5,
    'Rh': 6.0, 'Ta': 4.5, 'Pb': 4.0, 'Al': 3.5, 'C': 3.0, 'Hg': 0.6,
    'Pt': 0, 'Na': -2.0, 'K': -9.0, 'Ni': -15, 'Bi': -72
}
mpds_seebeck = {}

file = "/root/projects/ml-playground/data_prepearing/seebeck_coefficient_and_structure/data/PER_REV_AB_INITIO_not_empty_columns.xlsx"
data = pd.read_excel(file)
seebeck = data["Seebeck coefficient"].to_list()
formulas = data["Formula"].to_list()

for i, formula in enumerate(formulas):
    for key in seebeck_coefficients.keys():
        if key == formula:
            if key in mpds_seebeck:
                mpds_seebeck[key].append(seebeck[i])
            else:
                mpds_seebeck[key] = [seebeck[i]]

comparison = pd.DataFrame(columns=["Material", "MPDS", "Wiki"])

for key in mpds_seebeck.keys():
    new_row = {"Material": key,
               "MPDS": (mpds_seebeck[key][0] - mpds_seebeck['Pt'][0]),
               "Wiki": seebeck_coefficients[key]}
    comparison = comparison.append(new_row, ignore_index=True)

comparison.to_excel("/root/projects/ml-playground/data_prepearing/seebeck_coefficient_and_structure/data/comparison_wiki.xlsx", index=False)

