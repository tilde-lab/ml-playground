from data_massage.data_handler import DataHandler

api_key = 'KEY'
file_path = '/root/projects/ml-playground/data_massage/seebeck_coefficient_and_structure/data/26_3/'
file_name = 'ordered_str_1000'

if __name__ == "__main__":
    handler = DataHandler(True, api_key)

    # get Seebeck for PEER_REV and AB_INITIO
    seebeck_dfrm = handler.data_distributor(subject_of_request=0, max_value=1000, min_value=-1000)
    phases = set(seebeck_dfrm['Phase'].tolist())

    # get structure and make it ordered
    structures_dfrm = handler.data_distributor(subject_of_request=3, phases=phases)
    result_dfrm = handler.add_seebeck_by_phase_id(seebeck_dfrm, structures_dfrm)

    excel_file_path = file_path + file_name + ".xlsx"
    result_dfrm.to_excel(excel_file_path, index=False)
