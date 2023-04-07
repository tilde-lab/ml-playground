"""
	Predicting Seebeck Coefficient of ZnO using Decision tree regression model by
	extraction of ZnO structure from Materials Project platform using API 
	assuming decision.onnx, Utils.py files is already present in the directory
	
"""
import sys
from mp_api.client import MPRester
from pymatgen.io.ase import AseAtomsAdaptor
import onnx
import numpy
import onnxruntime as rt

sys.path.append("../")
from utils import get_APF, get_Wiener, get_Randic, get_R2score

# Retrieval of ZnO structure
with MPRester("your_api_key_here") as mpr:
    structure = mpr.get_structure_by_material_id("mp-2133")

# For conversion of Pymatgen structure to ase atoms
structure_ase = AseAtomsAdaptor.get_atoms(structure)

# Loading onnx model
onnx_model = onnx.load("decision.onnx")
onnx.checker.check_model(onnx_model)

# Running Inference session for predictions
sess = rt.InferenceSession("decision.onnx", providers=rt.get_available_providers())

# inputs of the model (in this case 2 inputs)
input_name1 = sess.get_inputs()[0].name
input_shape1 = sess.get_inputs()[0].shape
input_type1 = sess.get_inputs()[0].type
input_name2 = sess.get_inputs()[1].name
input_shape2 = sess.get_inputs()[1].shape
input_type2 = sess.get_inputs()[1].type

# output of the model
output_name = sess.get_outputs()[0].name
output_shape = sess.get_outputs()[0].shape
output_type = sess.get_outputs()[0].type

# Inputs (descriptors) for prediction
x1 = get_APF(structure_ase)
x1 = x1.astype(numpy.float32)  # should be same type as input_type1
x1 = x1.reshape(1)  # should be same shape as input_shape1
x2 = get_Wiener(structure_ase)
x2 = x2.astype(numpy.float32)  # should be same type as input_type2
x2 = x2.reshape(1)  # should be same shape as input_shape2

prediction = sess.run([output_name], {input_name1: x1, input_name2: x2})
