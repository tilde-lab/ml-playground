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
        self.api_key = api_key

    def make_request(self, is_seebeck=False, is_structure_for_seebeck=False, all_prop_for_seebeck=False, phases=None):
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

        elif (
                is_structure_for_seebeck and
                (self.dtype == MPDSDataTypes.PEER_REVIEWED or self.dtype == MPDSDataTypes.MACHINE_LEARNING)
        ):
            # change flag on PEER_REVIEWED for getting structure
            if self.dtype == MPDSDataTypes.MACHINE_LEARNING:
                self.client = MPDSDataRetrieval(dtype=MPDSDataTypes.PEER_REVIEWED, api_key=self.api_key)

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

        elif all_prop_for_seebeck == True:
            new_df = pd.DataFrame(phases, columns=["phase_id"])

            with open('/root/projects/ml-playground/data_massage/props.json') as f:
                props = eval(f.read())

            self.dtype = MPDSDataTypes.ALL

            for prop in list(props.keys()):
                try:
                    answer = self.client.get_data(
                        {"props": prop},
                        phases=phases
                    )

                    answers_lists = []
                    for value in answer:
                        # if value of specific prop not None
                        if value[6] != None:
                            update_values = [value[0], value[6]]
                            answers_lists.append(update_values)
                    answer_df = pd.DataFrame(answers_lists, columns=['phase_id', props[prop]])

                    mask = ~answer_df['phase_id'].duplicated()
                    answer_df_unique_phase_id = answer_df[mask]
                    new_df = pd.merge(answer_df_unique_phase_id, new_df, on='phase_id', how='outer')
                    if answers_lists != []:
                        print(f'Success! Save {len(answers_lists)} values for prop - {prop}')
                    else:
                        print(f'All values for prop: {prop} is None')
                except:
                    print(f'No data for {prop}')
            return new_df




