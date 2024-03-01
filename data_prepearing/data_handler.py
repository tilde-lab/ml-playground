from mpds_client import MPDSDataTypes
import pandas as pd
from database_handlers.MPDS.request_to_mpds import RequestMPDS
from data_prepearing.calculate_median_value import calc_median_value

class DataHandler:
    """
    Receiving and processing data.
    Implemented support for processing data just from MPDS.
    """
    def __init__(self, is_MPDS: bool, api_key: str, dtype=MPDSDataTypes.PEER_REVIEWED):
        """
        Initializes the client to access database.
        Implemented support for MPDS database.
        Parameters
        ----------
        is_MPDS : bool
            Parm is True, if you want to use MPDS
        api_key : str
            Key from your MPDS-account
        dtype : object from MPDSDataTypes
            Indicates the type of data being requested
        """
        if is_MPDS:
            self.client_handler = RequestMPDS(dtype=dtype, api_key=api_key)
        else:
            self.client_handler = None
        self.dtype = dtype

    def data_distributor(
            self, subject_of_request: int, max_value: int, min_value: int, is_uniq_phase_id=True,
            is_median_data=False, is_uniq_phase_to_many_props=False
    ):
        """
        Distributes queries according to the name of data required.
        Parameters
        ----------
        subject_of_request : int
            The number that corresponds to the name of the requested data
            Number 1: structures with calculation of Seebeck coefficient
        max_value : int
            Max value for range required data
            For subject_of_request=1 this is the Seebeck value
        min_value : int
            Min value for range required data
        is_uniq_phase_id : bool, optional
            Affects the filtering out of data with duplicate 'phase_id',
            if parm is True data will contain only a unique 'phase_id'
        is_median_data : bool, optional
            Calculate the median value for each example within specific 'phase_id'
        is_uniq_phase_to_many_props : bool, optional
            ! Supported just for subject_of_request=1
            Saves all structures for specific 'phase_id', while the Seebeck coefficient
            is only one for a specific 'phase_id'.
        """

        if subject_of_request == 1:
            result = self.seebeck_and_structure(
                max_value, min_value, is_uniq_phase_id, is_median_data, is_uniq_phase_to_many_props
            )

        return result

    def just_uniq_phase_id(self, df):
        """
        Saves one example for a specific 'phase_id', deletes subsequent ones.
        """
        try:
            mask = ~df['Phase'].duplicated()
        except:
            mask = ~df['phase_id'].duplicated()
        result_df = df[mask]
        return result_df

    def cleaning_trash_data(self, df, idx_check=6, type_of_trash=[]):
        """
        Deletes data with wrong information or empty data.
        """
        data = df.values.tolist()
        data = [row for row in data if row[idx_check] != type_of_trash]
        data = pd.DataFrame(
            data,
            columns=["phase_id", "entry", "cell_abc", "sg_n", "basis_noneq",
                     "els_noneq", "Formula", "Seebeck coefficient"]
        )
        return data

    def seebeck_and_structure(
            self, max_value, min_value, is_uniq_phase_id, is_median_data, is_uniq_phase_to_many_props
    ):
        """
        Requests all items that have a Seebeck coefficient. Queries atomic structures for all 'phase_id',
        which have a Seebeck coefficient. Cleans up invalid and empty data.
        Return pandas DataFrame.
        Parameters
        ----------
        max_value : int
            Max value for Seebeck coefficient
        min_value : int
            Min value for Seebeck coefficient
        is_uniq_phase_id : bool, optional
            Affects the filtering out of data with duplicate 'phase_id',
            if 'is_uniq_phase_id' is True data will contain only a unique 'phase_id'
        is_median_data : bool, optional
            Calculate the median value for each example within specific 'phase_id'
        is_uniq_phase_to_many_props : bool, optional
            Saves all structures for specific 'phase_id', while the Seebeck coefficient
            is only one for a specific 'phase_id'.
        """
        if is_median_data:
            is_uniq_phase_id = False

        dfrm = self.client_handler.make_request(is_seebeck=True)

        # remove outliers in value of Seebeck coefficient
        for index, row in dfrm.iterrows():
            if max_value < row['Seebeck coefficient'] or row['Seebeck coefficient'] < min_value:
                dfrm.drop(index, inplace=True)

        # leave only one phase_id value
        if is_uniq_phase_id:
            dfrm = self.just_uniq_phase_id(dfrm)

        phases = set(dfrm['Phase'].tolist())

        # get structures for data with Seebeck coefficient
        answer_df = self.client_handler.make_request(is_structure_for_seebeck=True, phases=phases)

        if is_uniq_phase_to_many_props:
            is_uniq_phase_id = False

        if is_uniq_phase_id:
            answer_df = self.just_uniq_phase_id(answer_df)

        dfrm.rename(columns={'Phase': 'phase_id'}, inplace=True)
        data = pd.merge(answer_df, dfrm, on='phase_id', how='inner')

        # remove empty data, make refactoring
        result = self.cleaning_trash_data(data, 6)
        result = self.cleaning_trash_data(result, 4)
        result = self.data_structure_refactoring(result)

        if is_median_data:
            result = calc_median_value(result, subject_of_request=1)

        return result

    def data_structure_refactoring(self, data):
        new_df = data.copy()
        new_df['entry'] = data['Formula']
        new_df['cell_abc'] = data['Seebeck coefficient']
        new_df['sg_n'] = data['entry']
        new_df['basis_noneq'] = data['cell_abc']
        new_df['els_noneq'] = data['sg_n']
        new_df['Formula'] = data['basis_noneq']
        new_df['Seebeck coefficient'] = data['els_noneq']
        new_df = new_df.rename(
            columns={'entry': 'Formula', 'cell_abc': 'Seebeck coefficient', 'sg_n': 'basis_noneq',
                     'basis_noneq': 'cell_abc', 'els_noneq': 'sg_n', 'Formula': 'basis_noneq',
                     'Seebeck coefficient': 'els_noneq'}
        )
        return new_df

    def combine_data(self, data_f, data_s):
        """  Simply connects 2 dataframes  """
        combined_df = pd.concat([data_f, data_s])
        return combined_df





