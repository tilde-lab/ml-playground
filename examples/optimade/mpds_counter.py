from mpds_client import MPDSDataRetrieval, MPDSDataTypes
import re
from itertools import groupby

api_key = 'KEY'
els = ["C", "Si", "Ge", "Sn", "Pb"]
groups = [0, 0, 0]
answers = []

client = MPDSDataRetrieval(dtype=MPDSDataTypes.ALL, api_key=api_key)
for el in els:
    answer = client.get_data(
        {"elements": el, "props": "atomic structure"}
    )
    [answers.append(row) for row in answer]

answers = [el for el, _ in groupby(answers)]

for idx in range(len(groups)):
    # elements HAS ANY "C","Si","Ge","Sn","Pb"
    if idx == 0:
        groups[idx] += len(answers)
    # elements HAS ANY "C",
    # "Si","Ge","Sn","Pb" AND nelements=2
    elif idx == 1:
        for row in answers:
            num_els = 0
            formula = row[1]
            formula_split = re.findall(r'[A-Z]?[^A-Z]*', formula)[:-1]
            for e in formula_split:
                num = re.findall(r'\d+', e)
                if num != [] :
                    num_els += int(num[0])
                else:
                    num_els += 1
            if num_els == 2:
                groups[idx] += 1
    # HAS ANY "C",
    # "Si","Ge","Sn" AND NOT elements HAS "Pb" AND
    # elements LENGTH 3
    else:
        for row in answers:
            num_els = 0
            formula = row[1]
            formula_split = re.findall(r'[A-Z]?[^A-Z]*', formula)[:-1]
            for e in formula_split:
                num = re.findall(r'\d+', e)
                if num != [] and 'Pb' not in formula:
                    num_els += int(num[0])
                elif 'Pb' not in formula:
                    num_els += 1
            if num_els == 3:
                groups[idx] += 1

print(groups)

