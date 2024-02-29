from mpds_client import MPDSDataRetrieval
import numpy as np
import pandas as pd

class RequestMPDS:
    """
    Makes requests to MPDS database.
    """

    def __init__(self, dtype, api_key):
        self.client = MPDSDataRetrieval(dtype=dtype, api_key=api_key)
        self.client.chillouttime = 1

    def make_request(self, is_seebeck=False, is_structure_for_seebeck=False, phases=None):
        """
        Requests data from the MPDS according to the input parms.
        Return DataFrame.
        """
        if is_seebeck:
            dfrm = self.client.get_dataframe({'props': 'Seebeck coefficient'})
            dfrm = dfrm[np.isfinite(dfrm['Phase'])]
            dfrm.rename(columns={'Value': 'Seebeck coefficient'}, inplace=True)
            dfrm.drop(dfrm.columns[[2, 3, 4, 5]], axis=1, inplace=True)
            return dfrm

        elif is_structure_for_seebeck:
            # get structures for data with Seebeck coefficient
            answer = self.client.get_data(
                {"props": "atomic structure"},
                phases=phases,
                fields={'S': ["phase_id", "entry", "cell_abc", "sg_n", "basis_noneq", "els_noneq"]}
            )
            answer_df = pd.DataFrame(answer, columns=["phase_id", "entry", "cell_abc", "sg_n", "basis_noneq", "els_noneq"])
            return answer_df




