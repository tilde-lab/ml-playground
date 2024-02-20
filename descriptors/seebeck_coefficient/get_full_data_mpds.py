import json

from mpds_client import MPDSDataTypes, MPDSDataRetrieval


def collect_data_in_dict(answer: dict, result):
    """
    saves data in form of dictionary, as follows:
    {formula: {phase_id: {properties}}}
    """
    for data in answer:
        if data != []:
            key = data[1]
            # key must be a string
            phase_id = str(data[0])

            if phase_id not in save_phase_id:
                save_phase_id.append(phase_id)

            if key not in result:
                result[key] = {phase_id: {data[4]: data[6]}}
            else:
                if phase_id in result[key]:
                    result[key][phase_id][data[4]] = data[6]
                else:
                    result[key][phase_id] = {}
                    result[key][phase_id][data[4]] = data[6]

    print(f"Number of different phases: {len(save_phase_id)}")

def save_to_json(file_path: str, file_name: str, data):
    with open((file_path + file_name), "w") as file:
        file.write(str(data))

api_key = "..."  # insert your key here...
file_result = "PEER_REVIEWED_AB_INITIO_all_data.json"
file_path = '/Users/alina/PycharmProjects/ml-playground/examples/'

save_phase_id = []
result = {}

client = MPDSDataRetrieval(api_key=api_key)
client.dtype = MPDSDataTypes.PEER_REVIEWED
client.dtype = MPDSDataTypes.AB_INITIO

with open("/Users/alina/PycharmProjects/ml-playground/examples/PEER_REVIEWED_AB_INITIO_seebeck.json", "r") as my_file:
    data_json = my_file.read()

result = json.loads(data_json)
total_result = result

structures = result.keys()

print(f'Total structures: {len(structures)}')

for cnt, formula in enumerate(structures):
    print(f'Step {cnt} from {len(structures)}')
    try:
        phase_id = [int(i) for i in list(result[formula].keys())]
        answer_for_specific_struct = client.get_data(
            search={"formulae": formula}, phases=phase_id
        )
        collect_data_in_dict(answer_for_specific_struct, total_result)
    except:
        phase_id = list(result[formula].keys())
        print(f'Failed to request data for {formula}, {phase_id}')
    if cnt % 50 == 0:
        print(f'------------SAVE data for {cnt} structures------------')

save_to_json(file_path, file_result, data=total_result)
