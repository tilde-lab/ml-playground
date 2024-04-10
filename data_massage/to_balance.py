import pandas as pd
import smogn

X = pd.read_csv(
    '/data_massage/seebeck_coefficient_and_structure/data/01_04/under_str_200_3.csv'
)
y = pd.read_csv(
    '/data_massage/seebeck_coefficient_and_structure/data/01_04/under_seeb_200_3.csv'
)

atoms = [eval(i) for i in X['atom'].values.tolist()]
distance = [eval(i) for i in X['distance'].values.tolist()]
total = []

for i in range(len(atoms)):
    total.append(atoms[i] + distance[i] + y.values.tolist()[i])

total_df = pd.DataFrame(total)
total_df.rename(columns={total_df.columns[-1]: 'Seebeck coefficient'}, inplace=True)

new_data = smogn.smoter(
    data=total_df,
    y='Seebeck coefficient',
    rel_method='auto'
)

new_l = []
seebeck_l = []

for row in new_data.values.tolist():
    atoms = row[:100]
    distance = row[100:200]
    seeb = row[-1]
    new_l.append([atoms, distance])
    seebeck_l.append(seeb)

df = pd.DataFrame(new_l, columns=['atom', 'distance'])
df_seeb = pd.DataFrame(seebeck_l, columns=['Seebeck coefficient'])

df.to_csv('/root/projects/ml-playground/data_massage/seebeck_coefficient_and_structure/data/01_04/over_str.csv', index=False)
df_seeb.to_csv('/root/projects/ml-playground/data_massage/seebeck_coefficient_and_structure/data/01_04/over_seeb.csv', index=False)

