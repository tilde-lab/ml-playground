from mpds_client import MPDSDataTypes
import pandas as pd
from pandas import DataFrame
from database_handlers.MPDS.request_to_mpds import RequestMPDS
from data_massage.calculate_median_value import calc_median_value
from ase import Atoms

# change path if another
from metis_backend.metis_backend.structures.struct_utils import order_disordered

class RequestTypes(object):
    JUST_SEEBECK = 0
    STRUCTURE_AND_SEEBECK = 1
    ALL_DATA_FOR_PHASES_WITH_SEEBECK = 2
    TO_ORDER_DISORDERED = 3

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
        self.available_dtypes = [1, 4]
        self.dtype = dtype
        self.api_key = api_key

    def data_distributor(
            self, subject_of_request: int, max_value=None, min_value=None, is_uniq_phase_id=True,
            is_median_data=False, is_uniq_phase_to_many_props=False, phases=None, data=None
    ):
        """
        Distributes queries according to the name of data required.
        Parameters
        ----------
        subject_of_request : int
            The number that corresponds to the name of the requested data
            Number 1: structures with calculation of Seebeck coefficient
            Number 2: all available props for specific phases with calculations of Seebeck
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
        if subject_of_request == 0:
            result = self.just_seebeck(
                max_value, min_value, is_uniq_phase_id
            )

        elif subject_of_request == 1:
            result = self.seebeck_and_structure(
                max_value, min_value, is_uniq_phase_id, is_median_data, is_uniq_phase_to_many_props
            )

        elif subject_of_request == 2:
            result = self.get_all_available_props(phases, data)

        elif subject_of_request == 3:
            result = self.to_order_disordered_str(phases)

        return result

    def just_seebeck(self, max_value: int, min_value: int, is_uniq_phase_id: bool) -> DataFrame:
        """
        Get Seebeck coefficient from db.
        """
        res_dfrm = pd.DataFrame(
            columns=['Phase', 'Formula', 'Seebeck coefficient']
        )

        for data_type in self.available_dtypes:
            self.client_handler = RequestMPDS(dtype=data_type, api_key=self.api_key)

            dfrm = self.client_handler.make_request(is_seebeck=True)

            # remove outliers in value of Seebeck coefficient
            if max_value != None:
                for index, row in dfrm.iterrows():
                    if max_value < row['Seebeck coefficient'] or row['Seebeck coefficient'] < min_value:
                        dfrm.drop(index, inplace=True)

            res_dfrm = self.combine_data(dfrm, res_dfrm)

        # leave only one phase_id value
        if is_uniq_phase_id:
           res_dfrm = self.just_uniq_phase_id(res_dfrm)

        return res_dfrm

    def to_order_disordered_str(self, phases: list):
        """
        Creates order in disordered structures.
        Returns pandas Dataframe with ordered structures.
        """
        # get disordered structures from db, save random structure for specific 'phase_id'
        all_data_df = self.just_uniq_phase_id(
            self.cleaning_trash_data(
                self.client_handler.make_request(is_structure=True, phases=phases), idx_check=5
            )
        )

        disordered_str = []
        for atomic_str in all_data_df.values.tolist():
            if atomic_str and any([occ != 1 for occ in atomic_str[1]]):
                disordered_str.append(atomic_str)

        disordered_df = pd.DataFrame(
            disordered_str,
            columns=['phase_id', 'occs_noneq', 'cell_abc', 'sg_n', 'basis_noneq', 'els_noneq']
        )

        result_list = []
        atoms_list = []

        # create Atoms objects
        for index, row in disordered_df.iterrows():
            # info for Atoms obj
            disordered = {'disordered': {}}

            basis_noneq = row['basis_noneq']
            els_noneq = row['els_noneq']
            occs_noneq = row['occs_noneq']

            for idx, (position, element, occupancy) in enumerate(zip(basis_noneq, els_noneq, occs_noneq)):
                # Add information about disorder to dict
                disordered['disordered'][idx] = {element: occupancy}
            crystal = Atoms(
                symbols=row['els_noneq'], positions=row['basis_noneq'], cell=row['cell_abc'], info=disordered
            )
            atoms_list.append(crystal)

        # make ordered structures
        for i, crystal in enumerate(atoms_list):
            obj, error = order_disordered(crystal)
            if not error:
                result_list.append([disordered_df['phase_id'].tolist()[i], obj.get_cell_lengths_and_angles().tolist(),
                                    disordered_df['sg_n'].tolist()[i],
                                    obj.get_positions().tolist(), list(obj.symbols)])
            else:
                print(error)

        new_ordered_df = pd.DataFrame(
            result_list,
            columns=['phase_id', 'cell_abc', 'sg_n', 'basis_noneq', 'els_noneq']
        )

        result_df = self.change_disord_on_ord(all_data_df.values.tolist(), new_ordered_df.values.tolist())

        return result_df

    def add_seebeck_by_phase_id(self, seebeck_df, structures_df) -> DataFrame:
        seebeck_df = seebeck_df.rename(columns={'Phase':'phase_id'})
        dfrm = pd.merge(seebeck_df, structures_df, on='phase_id', how='inner')
        return dfrm

    def just_uniq_phase_id(self, df: DataFrame) -> DataFrame:
        """
        Saves one example for a specific 'phase_id', deletes subsequent ones.
        """
        try:
            mask = ~df['Phase'].duplicated()
        except:
            mask = ~df['phase_id'].duplicated()
        result_df = df[mask]
        return result_df

    def cleaning_trash_data(self, df: DataFrame, idx_check=5, type_of_trash=[]) -> DataFrame:
        """
        Deletes data with wrong information or empty data.
        """
        data = df.values.tolist()
        data_res = []

        for row in data:
            if row[idx_check] != type_of_trash and row[idx_check] != None:
                data_res.append(row)
            else:
                print('Removed garbage data:', row)
        data = pd.DataFrame(
            data_res,
            columns=["phase_id", "occs_noneq", "cell_abc", "sg_n", "basis_noneq", "els_noneq"]
        )
        return data

    def get_all_available_props(self, phases: list, data):
        """
        Requests all available props for specific phases. Merge input Dataframe with new Dataframe with props.
        Parameters
        ----------
        phases : list
            'Phase_id' for request
        data : DataFrame
            DataFrame which will merge with result of request. Merge by 'inner'
        """
        props_df = self.client_handler.make_request(phases=phases, all_prop_for_seebeck=True)
        result_df = pd.merge(props_df, data, on='phase_id', how='inner')
        return result_df

    def seebeck_and_structure(
            self, max_value: int, min_value: int, is_uniq_phase_id: bool, is_median_data: bool,
            is_uniq_phase_to_many_props: bool
    ) -> DataFrame:
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

        # get seebeck
        dfrm = self.just_seebeck(max_value, min_value, is_uniq_phase_id)

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

    def data_structure_refactoring(self, data: DataFrame) -> DataFrame:
        new_df = data.copy()
        new_df['entry'] = data['Formula']
        new_df['cell_abc'] = data['Seebeck coefficient']
        new_df['sg_n'] = data['entry']
        new_df['basis_noneq'] = data['cell_abc']
        new_df['els_noneq'] = data['sg_n']
        new_df['Formula'] = data['basis_noneq']
        new_df['Seebeck coefficient'] = data['els_noneq']
        new_df.columns = ["phase_id", 'Formula', 'Seebeck coefficient', 'entry',
                          'cell_abc', 'sg_n', 'basis_noneq', 'els_noneq']

        return new_df

    def combine_data(self, data_f: DataFrame, data_s: DataFrame) -> DataFrame:
        """  Simply connects 2 dataframes  """
        combined_df = pd.concat([data_f, data_s])
        return combined_df

    def change_disord_on_ord(self, data_disord: list, ordered: list) -> DataFrame:
        """
        Create DataFrame with updated ordered values for disordered data.
        Other structures copy to new list without changes.
        Parameters
        ----------
        data_disord : list
            Made of 'phase_id', 'occs_noneq', 'cell_abc', 'sg_n', 'basis_noneq', 'els_noneq'
        ordered : list
            Made of next columns 'phase_id', 'cell_abc', 'sg_n', 'basis_noneq', 'els_noneq'
        """
        update_data = []
        loss_str = 0

        for dis_sample in data_disord:
            for i, ord_sample in enumerate(ordered):
                if dis_sample[0] == ord_sample[0]:
                    update_data.append(ord_sample)
                    break
                elif i == len(ordered) - 1:
                    # check that data is really sorted
                    if not (any([occ != 1 for occ in dis_sample[1]])):
                        update_data.append([dis_sample[0], dis_sample[2], dis_sample[3], dis_sample[4], dis_sample[5]])
                    else:
                        # see errors occurred in 'to_order_disordered_str'
                        loss_str += 1
                        print(f'Missing {loss_str} structures that could not pass ordering')

        dfrm = pd.DataFrame(update_data, columns=['phase_id', 'cell_abc', 'sg_n', 'basis_noneq', 'els_noneq'])
        return dfrm

    def removing_properties_by_intersection(self, data, props_to_save: list) -> DataFrame:
        columns_to_save = [
            'phase_id', 'Formula', 'Seebeck coefficient', 'entry', 'cell_abc', 'sg_n', 'basis_noneq', 'els_noneq'
        ]
        columns_to_save = columns_to_save + props_to_save

        new_df = data[columns_to_save].copy()
        new_df = new_df.dropna()
        return new_df



