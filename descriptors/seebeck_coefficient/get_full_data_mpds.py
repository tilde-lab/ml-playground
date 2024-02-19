from mpds_client import MPDSDataTypes
from mpds_client import MPDSDataRetrieval
import json

api_key = r"xr8qTrKa8E0363VhCTX8TIFuVXQisAE8tXUDqs0TvyQTLBTJ"
file_thermoelectric_power = "just_thermoelectric_power.json"
file_result = "data_seebeck.json"
file_path = "/root/projects/ml-playground/descriptors/thermoelectric_power/"

save_phase_id = []
result = {}

client = MPDSDataRetrieval(api_key=api_key)
client.dtype = MPDSDataTypes.ALL
client.dtype = MPDSDataTypes.MACHINE_LEARNING


def collect_data_in_dict(answer: list, dict_data):
    for data in answer:
        if data != []:
            key = data[1]
            phase_id = str(data[0])

            if key not in dict_data:
                continue
            else:
                if phase_id in dict_data[key]:
                    dict_data[key][phase_id][data[4]] = data[6]
                else:
                    continue

    return dict_data


def save_to_json(file_path: str, file_name: str, data):
    with open((file_path + file_name), "w") as file:
        file.write(str(data))


with open(
        "/root/projects/ml-playground/descriptors/thermoelectric_power/just_thermoelectric_power.json", "r"
) as my_file:
    data_json = my_file.read()

result = json.loads(data_json)
total_result = result.copy()

structures = result.keys()

for cnt, formula in enumerate(structures):
    [save_phase_id.append(int(i)) for i in list(result[formula].keys())]

print(f'Total structures: {len(structures)}')

for cnt, formula in enumerate(structures):
    print(f'Step {cnt} from {len(structures)}')
    try:
        phase_id = [int(i) for i in list(result[formula].keys())]
        answer_for_specific_struct = client.get_data(
            search={"formulae": formula}, phases=phase_id
        )
        total_result = collect_data_in_dict(answer_for_specific_struct, total_result)
    except:
        phase_id = list(result[formula].keys())
        print(f'Faileddd to request data for {formula}, {phase_id}')
    if cnt % 50 == 0:
        save_to_json(file_path, file_result, data=total_result)
        print(f'------------SAVE data for {cnt} structures------------')

save_to_json(file_path, file_result, data=total_result)