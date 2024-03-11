"""
Makes a request based on the existing “phase_id” (which has a Seebeck coefficient),
request occurs based on properties from "props.json" file. All characteristics for which at least 1 value != None
was found are saved.
If you want to leave only those props that are the majority,
use the function 'removing_properties_by_intersection' from the module data_massage.data_handler.
"""
from data_massage.data_handler import DataHandler, RequestTypes
from mpds_client import MPDSDataTypes
import pandas as pd

api_key = 'KEY'
data_type = MPDSDataTypes.ALL
subject_of_request = RequestTypes.ALL_DATA_FOR_PHASES_WITH_SEEBECK

file_path = '/root/projects/ml-playground/data_massage/seebeck_coefficient_and_structure/data/'
file_name = 'K_I_C_B_prop_ALL'
scaler_path = '/root/projects/ml-playground/data_massage/normalization/scalers/'

if __name__ == "__main__":
    handler = DataHandler(True, api_key, data_type)

    file = "/data_massage/seebeck_coefficient_and_structure/data/PEER_REV_AB_INITIO_not_empty_columns.xlsx"
    data = pd.read_excel(file)
    phases = set(data['phase_id'].tolist())

    result = handler.data_distributor(
        subject_of_request=subject_of_request, phases=phases, data=data
    )

    excel_file_path = file_path + file_name + ".xlsx"
    result.to_excel(excel_file_path, index=False)