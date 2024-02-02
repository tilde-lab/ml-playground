"""
------------
Predicts Seebeck Coefficient of a given crystalline structure. 
Requires 2 command-line arguments.
	Arguments:
		Structure file: CIF or POSCAR format
		Machine Learning model: ONNX format
	Returns:
		Predicted Value of Seebeck Coefficient
------------
"""
import sys
import onnx
import numpy
import onnxruntime as rt
from struct_utils import (
    detect_format,
    poscar_to_ase,
    optimade_to_ase,
    refine,
)
from cif_utils import cif_to_ase

sys.path.append("../")
from descriptors.utils import get_APF, get_Wiener

# Printing Docstring information for wrong number of command-line arguments
if len(sys.argv) != 3:
    print(__doc__)

# Extracting ASE object from structure file
structure = open(sys.argv[1]).read()
fmt = detect_format(structure)

if fmt == "cif":
    ase_obj, error = cif_to_ase(structure)
    if error:
        raise RuntimeError(error)

elif fmt == "poscar":
    ase_obj, error = poscar_to_ase(structure)
    if error:
        raise RuntimeError(error)

elif fmt == "optimade":
    ase_obj, error = optimade_to_ase(structure, skip_disorder=True)
    if error:
        raise RuntimeError(error)

else:
    raise RuntimeError("Provided data format unsuitable or not recognized")

ase_obj, error = refine(ase_obj, conventional_cell=True)
if error:
    raise RuntimeError(error)

# Loading the ONNX model
model = sys.argv[2]
onnx_model = onnx.load(model)
onnx.checker.check_model(onnx_model)

# Running Inference session for predictions
sess = rt.InferenceSession(model, providers=rt.get_available_providers())

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
x1 = get_APF(ase_obj)
x1 = x1.astype(numpy.float32)
x1 = x1.reshape(input_shape1)
x2 = get_Wiener(ase_obj)
x2 = x2.astype(numpy.float32)
x2 = x2.reshape(input_shape2)

prediction = sess.run([output_name], {input_name1: x1, input_name2: x2})
print(prediction)
