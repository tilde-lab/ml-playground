from mpds_client import MPDSDataTypes
from mpds_client import MPDSDataRetrieval
import numpy as np
import pandas as pd

api_key = r"KEY"
file_name = "4_processed_structure_and_seebeck"
file_path = '/Users/alina/PycharmProjects/ml-playground/examples/'

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

    # delete rows with empty 'basis_noneq'
    for index, row in answer_df.iterrows():
        if row['basis_noneq'] == None:
            answer_df.drop(index, inplace=True)

    dfrm.rename(columns={'Phase': 'phase_id'}, inplace=True)

    # merge only those data by 'phase_id' for which there is a structure
    merged_df = pd.merge(dfrm, answer_df, on='phase_id', how='inner')

    excel_file_path = file_path + file_name + ".xlsx"
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
    dfrm.drop(dfrm.columns[[2, 3, 4, 5]], axis=1, inplace=True)

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

    # delete rows with empty 'basis_noneq'
    for index, row in answer_df.iterrows():
        if row['basis_noneq'] == None:
            answer_df.drop(index, inplace=True)

    mask = ~answer_df['phase_id'].duplicated()
    answer_df_unique_phase_id = answer_df[mask]

    dfrm_unique_phase_id.rename(columns={'Phase': 'phase_id'}, inplace=True)

    # merge only those data by 'phase_id' for which there is a structure
    merged_df = pd.merge(dfrm_unique_phase_id, answer_df_unique_phase_id, on='phase_id', how='inner')

    excel_file_path = file_path + file_name + ".xlsx"
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


excel_file_path = file_path + file_name + ".xlsx"
excel_file_path_result = file_path + 'total_data_repetitive' + ".xlsx"

get_seebeck_repetitive_phase(file_path, file_name)

# data = pd.read_excel(excel_file_path)
# get_other_props_not_repetitive(excel_file_path_result, data)