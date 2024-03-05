from data_prepearing.data_handler import DataHandler
from mpds_client import MPDSDataTypes
import random
from data_prepearing.normalization.normalization import make_normalization
import pandas as pd

api_key = 'KEY'
data_type = MPDSDataTypes.AB_INITIO
subject_of_request = 1
max_value = 1000
min_value = -1000
is_uniq_phase_id = True
is_median_data = False
is_uniq_phase_to_many_props = False
file_path = '/Users/alina/PycharmProjects/ml-playground/data_prepearing/seebeck_coefficient_and_structure/data/'
file_name = 'AB_INITIO_PEER_REV_uniq_phase'
scaler_path = '/Users/alina/PycharmProjects/ml-playground/data_prepearing/normalization/scalers/'

if __name__ == "__main__":
    handler = DataHandler(True, api_key, data_type)

    file1 = "/Users/alina/PycharmProjects/ml-playground/data_prepearing/seebeck_coefficient_and_structure/data/AB_INITIO_uniq_phase_id.xlsx"
    file2 = "/Users/alina/PycharmProjects/ml-playground/data_prepearing/seebeck_coefficient_and_structure/data/4_processed_structure_and_seebeck_uniq.xlsx"

    data1 = pd.read_excel(file1)
    data2 = pd.read_excel(file2)

    result = handler.combine_data(data1, data2)

    random_number = str(random.randint(100000, 999999))
    norm_result = make_normalization(
        result, scaler_path, scaler_name=('scaler' + random_number)
    )

    excel_file_path = file_path + file_name + random_number + ".xlsx"
    result.to_excel(excel_file_path, index=False)