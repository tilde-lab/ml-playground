"""
Get median Seebeck and structures.
1 phase_id <-> 1 Seebeck <-> many structures.
"""
from data_massage.data_handler import DataHandler
from data_massage.calculate_median_value import seebeck_median_value

api_key = 'KEY'
file_path = '/root/projects/ml-playground/data_massage/seebeck_coefficient_and_structure/data/26_3/'
file_name = 'median_rep_ordered_str_200'

if __name__ == "__main__":
    handler = DataHandler(True, api_key)

    # get Seebeck for PEER_REV and AB_INITIO
    seebeck_dfrm = handler.data_distributor(subject_of_request=0, max_value=200, min_value=-150, is_uniq_phase_id=False)
    phases = set(seebeck_dfrm['Phase'].tolist())
    # make median Seebeck value
    median_seebeck = seebeck_median_value(seebeck_dfrm, phases)

    # get structure and make it ordered
    structures_dfrm = handler.data_distributor(subject_of_request=3, phases=phases, is_uniq_phase_id=False)
    result_dfrm = handler.add_seebeck_by_phase_id(median_seebeck, structures_dfrm)

    csv_file_path = file_path + file_name + ".csv"
    result_dfrm.to_csv(csv_file_path, index=False)

