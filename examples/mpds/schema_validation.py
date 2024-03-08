from urllib.parse import urlencode
import pandas as pd
import sys
import json
import httplib2
from jsonschema import validate, Draft4Validator
from jsonschema.exceptions import ValidationError
import time

api_key = "KEY"
endpoint = "https://api.mpds.io/v0/download/facet"

file = "/Users/alina/PycharmProjects/ml-playground/my_tests_for_dev/data/PER_REV_AB_INITIO_not_empty_columns.xlsx"
data = pd.read_excel(file)
phases = data = set(data['phase_id'].tolist())
phases_str = ''

req = httplib2.Http()
network = httplib2.Http()

try:
    response, schema = network.request("http://developer.mpds.io/mpds.schema.json")
except:
    assert response.status == 200

schema = json.loads(schema)
Draft4Validator.check_schema(schema)

# use just 1000 phases (quantity condition for request)
for phase in range(1000):
    p = (list(phases))[phase]
    if phase != 999:
        phases_str += str(p) + ',' + ' '
    else:
        phases_str += str(p)

elements = ["Sr", "Zr", "Br-Cl", "Li-Mg-O", "H-O-Cl-Br", "Mg-Au"]
classes = ["perovskite", "oxide", "conductor"]
props = {'electron energy band structure': 'B', 'electron density of states': 'C',
             'electrical conductivity': 'D', 'isothermal bulk modulus': 'E', 'Young modulus': 'F',
             'shear modulus': 'G', 'poisson ratio': 'H', 'enthalpy of formation': 'I',
             'energy gap for direct transition': 'J', 'heat capacity at constant pressure': 'K', 'entropy': 'L',
             'vibrational spectra': 'M', 'Raman spectra': 'N', 'effective charge': 'O', 'infrared spectra': 'P',
             'energy gap for indirect transition': 'Q'}
    # , 'atomic structure': 'X'}

for prop in props.keys():
    for el in elements:
        for cl in classes:
            search = [{"props": prop}, {"props": prop, "elements": el, "classes": cl},
                      {"elements": el, "classes": cl, "props": prop}]
            for option in search:

                json_request = {
                        'q': json.dumps(option),
                        # 'phases' : phases_str,
                        'pagesize': 10,
                        'dtype': 7
                    }

                response, content = req.request(
                    uri=endpoint + '?' + urlencode(json_request),
                    method='GET',
                    headers={'Key': api_key}
                )
                time.sleep(1)

                target = json.loads(content)

                if response.status != 200:
                    raise RuntimeError("Error code %s" % response.status)
                if target.get('error'):
                    raise RuntimeError(target['error'])

                print("OK, got %s hits" % len(target['out']))

                if not target.get("npages") or not target.get("out") or target.get("error"):
                    print("Unexpected API response")
                    continue

                try:
                    validate(instance=target["out"], schema=schema)
                    print(f"Success for search: {option}")
                except ValidationError as e:
                    raise RuntimeError(
                        "The item: \r\n\r\n %s \r\n\r\n has an issue: \r\n\r\n %s" % (
                            e.instance, e.context
                        )
                    )


