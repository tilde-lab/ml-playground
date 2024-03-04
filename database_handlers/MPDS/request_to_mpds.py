from mpds_client import MPDSDataRetrieval
from mpds_client import MPDSDataTypes
import numpy as np
import pandas as pd
import httplib2
import json
import time

class RequestMPDS:
    """
    Makes requests to MPDS database.
    """

    def __init__(self, dtype, api_key=None):
        self.client = MPDSDataRetrieval(dtype=dtype, api_key=api_key)
        self.client.chillouttime = 1
        self.dtype = dtype

    def make_request(self, is_seebeck=False, is_structure_for_seebeck=False, phases=None):
        """
        Requests data from the MPDS according to the input parms.
        Return DataFrame or dict.
        """
        if is_seebeck:
            dfrm = self.client.get_dataframe({'props': 'Seebeck coefficient'})
            dfrm = dfrm[np.isfinite(dfrm['Phase'])]
            dfrm.rename(columns={'Value': 'Seebeck coefficient'}, inplace=True)
            dfrm.drop(dfrm.columns[[2, 3, 4, 5]], axis=1, inplace=True)
            return dfrm

        elif is_structure_for_seebeck and self.dtype == MPDSDataTypes.PEER_REVIEWED:
            # get structures for data with Seebeck coefficient
            answer = self.client.get_data(
                {"props": "atomic structure"},
                phases=phases,
                fields={'S': ["phase_id", "entry", "cell_abc", "sg_n", "basis_noneq", "els_noneq"]}
            )
            answer_df = pd.DataFrame(answer, columns=["phase_id", "entry", "cell_abc", "sg_n", "basis_noneq", "els_noneq"])
            return answer_df

        elif is_structure_for_seebeck and self.dtype == MPDSDataTypes.AB_INITIO:
            loss_data = 0
            results = []

            for phase in phases:
                query = 'https://api.mpds.io/v0/download/s?fmt=optimade&q=S500'

                req = httplib2.Http()
                response, content = req.request(query + str(phase))

                # if there is no data
                if response.status != 200:
                    loss_data += 1
                    print(f'No data found {loss_data} from {len(phases)}')
                    time.sleep(1)
                    continue
                else:
                    try:
                        answer = (json.loads(content))['data'][0]['attributes']
                    except:
                        loss_data += 1
                        print(f'INCORRECT data {loss_data}: ', content)
                        time.sleep(1)
                        continue
                    result_sample = [phase, answer["immutable_id"]]
                    result_sample.append(answer['lattice_vectors'])
                    result_sample.append(1)
                    result_sample.append(answer['cartesian_site_positions'])
                    result_sample.append(answer['species_at_sites'])

                    results.append(result_sample)
                    time.sleep(1)

            if results != []:
                print(f'Got {len(results)} hits')

            answer_df = pd.DataFrame(results,
                                     columns=["phase_id", "entry", "cell_abc", "sg_n", "basis_noneq", "els_noneq"])

            return answer_df




