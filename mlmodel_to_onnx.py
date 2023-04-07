# Converting coreml format to onnx format
import coremltools
import onnxmltools
import onnx

# Export the model created in Turi Create into CoreML format
model_decision.export_coreml("decision.mlmodel")

# Update the input name and path for your CoreML model
input_coreml_model = "decision.mlmodel"

# Change this path to the output name and path for the ONNX model
output_onnx_model = "decision.onnx"

# Load your CoreML model
coreml_model = coremltools.utils.load_spec(input_coreml_model)

# Convert the CoreML model into ONNX
onnx_model = onnxmltools.convert_coreml(coreml_model)

# Save as protobuf
onnxmltools.utils.save_model(onnx_model, output_onnx_model)

# Load onnx model and check
onnx_model = onnx.load("decision.onnx")
onnx.checker.check_model(onnx_model)
