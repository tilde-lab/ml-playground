from mpds_client import MPDSDataTypes
from mpds_client import MPDSDataRetrieval

api_key = r"xr8qTrKa8E0363VhCTX8TIFuVXQisAE8tXUDqs0TvyQTLBTJ"
file_thermoelectric_power = "just_thermoelectric_power.json"
file_path = "/home/alina/PycharmProjects/ml-playground/descriptors/seebeck_coefficient/data/"

save_phase_id = []
result = {}

client = MPDSDataRetrieval(api_key=api_key)
client.dtype = MPDSDataTypes.ALL
client.dtype = MPDSDataTypes.MACHINE_LEARNING

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

# get data with thermoelectric power coefficient
answer = client.get_data(
    {'props': 'thermoelectric power'}
)

collect_data_in_dict(answer, result)

with open((file_path + file_thermoelectric_power), "w") as file:
    file.write(str(result))

structures = result.keys()

print(f'Total structures: {len(structures)}')





