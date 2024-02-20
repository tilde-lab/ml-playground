import json

file_path = "/home/alina/PycharmProjects/ml-playground/descriptors/seebeck_coefficient/data/processed_compress_data_1/"
file_input = 'PEER_REVIEWED_AB_INITIO_all_data.json'
file_output = 'processed_PEER_REVIEWED_AB_INITIO.json'
file_keys = "keys_for_compress.json"

with open(
        ('/home/alina/PycharmProjects/ml-playground/descriptors/seebeck_coefficient/data/' + file_input), "r"
) as my_file:
    data_json = my_file.read()

data = json.loads(data_json)
compress_data = {}
original_keys = []
update_keys ={}
update_data = {}

# save all original keys
for formula in data:
    for phases in data[formula]:
        for key in list(data[formula][phases].keys()):
            if key not in original_keys:
                original_keys.append(key)
# compare new value to key
for cnt, key in enumerate(original_keys):
    update_keys[key] = str(chr(65 + cnt))

with open((file_path + file_keys), "w") as file:
    file.write(str(update_keys))

for formula in data:
    for phases in data[formula]:
        new_keys_dict = {}
        for key in data[formula][phases].keys():
            compress_key = update_keys[key]
            new_keys_dict[compress_key] = data[formula][phases][key]
        if formula not in update_data:
            update_data[formula] = {}
            update_data[formula][phases] = new_keys_dict
        else:
            update_data[formula][phases] = new_keys_dict

with open((file_path + file_output), "w") as file:
    file.write(str(update_data))