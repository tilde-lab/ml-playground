from data_massage.data_handler import DataHandler

api_key = 'KEY'
file_path = '/root/projects/ml-playground/data_massage/seebeck_coefficient_and_structure/data/26_3/'
file_name = 'rep_disordered_str_200'

if __name__ == "__main__":
    handler = DataHandler(True, api_key)

    # get Seebeck and structure (ordered + disordered)
    seebeck_structure_dfrm = handler.data_distributor(
        subject_of_request=1, max_value=200, min_value=-150, is_uniq_phase_id=False
    )

    excel_file_path = file_path + file_name + ".xlsx"
    seebeck_structure_dfrm.to_excel(excel_file_path, index=False)
