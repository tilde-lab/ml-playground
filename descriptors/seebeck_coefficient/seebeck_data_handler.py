import json

file_path = "/home/alina/PycharmProjects/ml-playground/descriptors/seebeck_coefficient/data/"
file_input = 'data_seebeck.json'
file_output = 'processed_data_seebeck.json'

with open((file_path + file_input), "r") as my_file:
    data_json = my_file.read()

data = json.loads(data_json)
update_data = {}

for formula in data:
    for phases in data[formula]:
        # there must be other properties besides Seebeck
        if len(list(data[formula][phases].keys())) < 2:
            continue
        else:
            if formula not in update_data:
                update_data[formula] = {}
                update_data[formula][phases] = data[formula][phases]
            else:
                update_data[formula][phases] = data[formula][phases]

print(len(list(update_data.keys())))

with open((file_path + file_output), "w") as file:
    file.write(str(data))