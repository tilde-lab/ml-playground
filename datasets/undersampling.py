import pandas as pd

str = pd.read_csv(
        '/Users/alina/PycharmProjects/ml-playground/data_massage/seebeck_coefficient_and_structure/data/01_04/vectors_str_200.csv'
)
seebeck = pd.read_csv(
    '/Users/alina/PycharmProjects/ml-playground/data_massage/seebeck_coefficient_and_structure/data/01_04/rep_seebeck_200.csv'
)
data = pd.concat([seebeck["Seebeck coefficient"], str], axis=1).values.tolist()

new_list_str = []
new_list_seeb = []
seeb_used = []

cnt = 0

for row in sorted(data):
    if float(round(row[0])) not in seeb_used:
        cnt = 1
        seeb_used.append(float(round(row[0])))
        new_list_seeb.append(float(round(row[0])))
        new_list_str.append([row[1], row[2]])
    else:
        if cnt >= 50:
            cnt += 1
            continue
        else:
            new_list_seeb.append(float(round(row[0])))
            new_list_str.append([row[1], row[2]])
            cnt += 1


pd.DataFrame(new_list_seeb, columns=['Seebeck coefficient']).to_csv(
    '/Users/alina/PycharmProjects/ml-playground/data_massage/seebeck_coefficient_and_structure/data/01_04/under_seeb_200.csv',
    index=False
)
pd.DataFrame(new_list_str, columns=['atom', 'distance']).to_csv(
    '/Users/alina/PycharmProjects/ml-playground/data_massage/seebeck_coefficient_and_structure/data/01_04/under_str_200.csv',
    index=False
)
