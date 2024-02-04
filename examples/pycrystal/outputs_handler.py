import os, sys
from pprint import pprint
import json

from pycrystal import CRYSTOUT

dir_path = "/home/alina/PycharmProjects/metis_first_not_bild/tasks/20240202_202821_120"
file = "OUTPUT"

path_input_file = (
    os.path.join(dir_path, file)
)

if not path_input_file or not os.path.exists(path_input_file):
    sys.exit("Incorrect path")

assert (CRYSTOUT.acceptable(path_input_file))

result = CRYSTOUT(path_input_file)
pprint(result.info)

