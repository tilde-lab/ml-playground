import pandas as pd

str = pd.read_csv(
    '/root/projects/ml-playground/data_massage/seebeck_coefficient_and_structure/data/01_04/vectors_str_200.csv'
)
seebeck = pd.read_csv(
    '/root/projects/ml-playground/data_massage/seebeck_coefficient_and_structure/data//01_04/rep_seebeck_200.csv'
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


seeb_df = pd.DataFrame(new_list_seeb, columns=['Seebeck coefficient'])
str_df = pd.DataFrame(new_list_str, columns=['atom', 'distance'])
shuffle_df = pd.concat([seeb_df, str_df], axis=1).sample(frac=1).reset_index(drop=True)

df_seebeck = shuffle_df.iloc[:,:1]
df_structs = shuffle_df.iloc[:,1:]

df_seebeck.to_csv('/root/projects/ml-playground/data_massage/seebeck_coefficient_and_structure/data/01_04/under_seeb_200.csv', index=False)
df_structs.to_csv('/root/projects/ml-playground/data_massage/seebeck_coefficient_and_structure/data/01_04/under_str_200.csv', index=False)
