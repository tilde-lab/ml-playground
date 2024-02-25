from mpds_client import MPDSDataTypes
from mpds_client import MPDSDataRetrieval
import numpy as np
import pandas as pd
import statistics

api_key = r"KEY"
file_name = "5_processed_structure_and_seebeck"
file_path = '/Users/alina/PycharmProjects/ml-playground/descriptors/seebeck_coefficient/data/'

client = MPDSDataRetrieval(dtype=MPDSDataTypes.PEER_REVIEWED, api_key=api_key)
client.chillouttime = 1

def get_seebeck_repetitive_phase(file_path, file_name):
    '''
    Stores all data with the Sеebeck coefficient.
    Requests structures, deletes those data for which structures are not found.
    '''
    dfrm = client.get_dataframe({'props': 'Seebeck coefficient'})
    dfrm = dfrm[np.isfinite(dfrm['Phase'])]

    dfrm.rename(columns={'Value': 'Seebeck coefficient'}, inplace=True)
    dfrm.drop(dfrm.columns[[2, 3, 4, 5]], axis=1, inplace=True)

    # remove outliers
    for index, row in dfrm.iterrows():
        if 1000 < row['Seebeck coefficient'] or row['Seebeck coefficient'] < -1000:
            dfrm.drop(index, inplace=True)

    phases = set(dfrm['Phase'].tolist())
    answer = client.get_data(
        {"props": "atomic structure"},
        phases=phases,
        fields={'S':["phase_id", "entry", "cell_abc", "sg_n", "basis_noneq", "els_noneq"]}
    )

    answer_df = pd.DataFrame(answer, columns=["phase_id", "entry", "cell_abc", "sg_n", "basis_noneq", "els_noneq"])

    for index, row in answer_df.iterrows():
        if row['basis_noneq'] == None or row['basis_noneq'] == '[]':
            answer_df.drop(index, inplace=True)

    dfrm.rename(columns={'Phase': 'phase_id'}, inplace=True)

    # merge only those data by 'phase_id' for which there is a structure
    merged_df = pd.merge(dfrm, answer_df, on='phase_id', how='inner')

    excel_file_path = file_path + file_name + ".xlsx"
    merged_df = merged_df[merged_df['basis_noneq'] != '[]']
    merged_df.to_excel(excel_file_path, index=False)


def get_seebeck_not_repetitive_phase(file_path, file_name):
    '''
    Stores data with unique phase_id.
    Requests structures, deletes those data for which structures are not found.
    Deletes structures with repeating phase_id.
    '''
    dfrm = client.get_dataframe(
        {'props': 'Seebeck coefficient'}
    )
    dfrm = dfrm[dfrm["Units"] == "muV K-1"]
    dfrm = dfrm[np.isfinite(dfrm['Phase'])]
    dfrm.rename(columns={'Value': 'Seebeck coefficient'}, inplace=True)
    dfrm.drop(dfrm.columns[[2, 4, 5]], axis=1, inplace=True)

    # remove outliers
    for index, row in dfrm.iterrows():
        if 1000 < row['Seebeck coefficient'] or row['Seebeck coefficient'] < -1000:
            dfrm.drop(index, inplace=True)

    # leave only one phase_id value
    mask = ~dfrm['Phase'].duplicated()
    dfrm_unique_phase_id = dfrm[mask]

    phases = set(dfrm_unique_phase_id['Phase'].tolist())

    answer = client.get_data(
        {"props": "atomic structure"},
        phases=phases,
        fields={'S': ["phase_id", "entry", "cell_abc", "sg_n", "basis_noneq", "els_noneq"]}
    )

    answer_df = pd.DataFrame(answer, columns=["phase_id", "entry", "cell_abc", "sg_n", "basis_noneq", "els_noneq"])

    for index, row in answer_df.iterrows():
        if row['basis_noneq'] == None or row['basis_noneq'] == '[]':
            answer_df.drop(index, inplace=True)

    mask = ~answer_df['phase_id'].duplicated()
    answer_df_unique_phase_id = answer_df[mask]

    dfrm_unique_phase_id.rename(columns={'Phase': 'phase_id'}, inplace=True)

    # merge only those data by 'phase_id' for which there is a structure
    merged_df = pd.merge(dfrm_unique_phase_id, answer_df_unique_phase_id, on='phase_id', how='inner')

    excel_file_path = file_path + file_name + ".xlsx"
    merged_df = merged_df[merged_df['basis_noneq'] != '[]']
    merged_df.to_excel(excel_file_path, index=False)

def get_other_props_not_repetitive(path_for_save_result, merged_df):
    '''
    Stores one value for each property for each phase_is.
    '''
    props = {'electron energy band structure': 'B', 'electron density of states': 'C',
             'electrical conductivity': 'D', 'isothermal bulk modulus': 'E', 'Young modulus': 'F',
             'shear modulus': 'G', 'poisson ratio': 'H', 'enthalpy of formation': 'I',
             'energy gap for direct transition': 'J', 'heat capacity at constant pressure': 'K', 'entropy': 'L',
             'vibrational spectra': 'M', 'Raman spectra': 'N', 'effective charge': 'O', 'infrared spectra': 'P',
             'energy gap for indirect transition': 'Q'}

    phases = set(merged_df['phase_id'].tolist())

    for prop in list(props.keys()):
        try:
            answer = client.get_data(
                {"props": prop},
                phases=phases
            )

            answers_lists = []
            for value in answer:
                update_values = [value[0], value[6]]
                answers_lists.append(update_values)
            answer_df = pd.DataFrame(answers_lists, columns=['phase_id', prop])

            mask = ~answer_df['phase_id'].duplicated()
            answer_df_unique_phase_id = answer_df[mask]

            merged_df = pd.merge(answer_df_unique_phase_id, merged_df, on='phase_id', how='outer')
            print('Succes!')
        except:
            print(f'No data for {prop}')

    merged_df.to_excel(path_for_save_result, index=False)

    print('Number of phases with structure and descriptors:', len(set(merged_df.iloc[:, 1])))

def make_median_value_for_phase_id():
    '''
    Calculates median Seebeck value for materials with same Phase_ids, calculates
    average coordinate value for each atom (at each coordinate) in materials with same phase_id
    '''
    excel_file_path = file_path + file_name + ".xlsx"
    data = pd.read_excel(excel_file_path)
    data = data.values.tolist()

    new_data_list = []

    phases = list(set([i[0] for i in data]))

    for phase in phases:
        seebeck = []
        x_y_z = []
        data_for_phase = [string for string in data if string[0] == phase]
        update_data_for_phase = [row for row in data_for_phase if row[6] != '[]']

        if update_data_for_phase == []:
            continue

        for value in update_data_for_phase:
            seebeck.append(value[2])
            x_y_z.append(eval(value[6]))

        # if different number of atoms in structure for specific phase_id,
        # consider median value for those cases of which there are more
        if len(set(len(sample) for sample in x_y_z)) > 1:
            count_len = {}
            for l in set(len(sample) for sample in x_y_z):
                for sample in x_y_z:
                    if len(sample) == l:
                        if str(l) in count_len:
                            count_len[str(l)] = count_len[str(l)] + 1
                        else:
                            count_len[str(l)] = 1
            often_len = int(max(count_len, key=count_len.get))

            update_data_for_phase = []

            # delete rows with different number of atoms compared to most cases
            for value in data_for_phase:
                if len(eval(value[6])) != often_len:
                    continue
                else:
                    update_data_for_phase.append(value)

            # repeat again for update data
            x_y_z = []
            seebeck = []
            for value in update_data_for_phase:
                seebeck.append(value[2])
                x_y_z.append(eval(value[6]))

        median_seebeck = statistics.median(seebeck)
        median_x_y_z = []

        for atom in range(len(x_y_z[0])):
            x, y, z = [], [], []
            for sample in x_y_z:
                x.append(sample[atom][0])
                y.append(sample[atom][1])
                z.append(sample[atom][2])
            x_median, y_median, z_median = statistics.median(x), statistics.median(y), statistics.median(z)
            median_x_y_z.append([x_median, y_median, z_median])

        new_data_for_phase = update_data_for_phase[0]
        new_data_for_phase[2] = median_seebeck
        new_data_for_phase[6] = str(median_x_y_z)

        new_data_list.append(new_data_for_phase)

    data = pd.DataFrame(new_data_list, columns=["phase_id", "Formula", "Seebeck coefficient", "entry", "cell_abc", "sg_n", "basis_noneq", "els_noneq"])
    excel_file_path_result = file_path + 'median_structure_and_seebeck' + ".xlsx"
    data.to_excel(excel_file_path_result, index=False)

# get_seebeck_repetitive_phase(file_path, file_name)
make_median_value_for_phase_id()

# excel_file_path = file_path + file_name + ".xlsx"
# excel_file_path_result = file_path + 'total_data_repetitive' + ".xlsx"
# data = pd.read_excel(excel_file_path)
# get_other_props_not_repetitive(excel_file_path_result, data)