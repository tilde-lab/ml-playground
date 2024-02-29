from data_prepearing.data_handler import DataHandler
from mpds_client import MPDSDataTypes
import random
from data_prepearing.normalization.normalization import make_normalization

api_key = 'KEY'
data_type = MPDSDataTypes.PEER_REVIEWED
subject_of_request = 1
max_value = 1000
min_value = -1000
is_uniq_phase_id = True
is_median_data = False
is_uniq_phase_to_many_props = False
file_path = '/Users/alina/PycharmProjects/ml-playground/data_prepearing/seebeck_coefficient_and_structure/data/'
file_name = 'example_uniq_phase_id_'
scaler_path = '/Users/alina/PycharmProjects/ml-playground/data_prepearing/normalization/scalers/'

if __name__ == "__main__":
    handler = DataHandler(True, api_key, data_type)
    result = handler.data_distributor(subject_of_request, max_value, min_value, is_uniq_phase_id,
                                      is_median_data=is_median_data,
                                      is_uniq_phase_to_many_props=is_uniq_phase_to_many_props
                                      )

    random_number = str(random.randint(100000, 999999))
    norm_result = make_normalization(
        result, scaler_path, scaler_name=('scaler' + random_number)
    )

    excel_file_path = file_path + file_name + random_number + ".xlsx"
    norm_result.to_excel(excel_file_path, index=False)