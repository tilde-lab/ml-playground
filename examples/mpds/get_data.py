from mpds_client import MPDSDataTypes
from mpds_client import MPDSDataRetrieval

api_key = r"KEY"

client = MPDSDataRetrieval(api_key=api_key)
client.dtype = MPDSDataTypes.ALL
client.dtype = MPDSDataTypes.MACHINE_LEARNING

answer = client.get_data(
    {"formulae": "H2O"}
)

result = {}
save_phase_id = []

for data in answer:
    if data != []:
        key = data[1]
        phase_id = data[0]

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

file_path = "/home/alina/PycharmProjects/ml-playground/examples/mpds/result/data"

with open(file_path, "w") as file:
    file.write(str(result))

for key in result.keys():
    print(f"{key} ------ {result[key]}")