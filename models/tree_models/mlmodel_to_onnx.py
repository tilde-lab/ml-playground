# Converting coreml format to onnx format
import coremltools
import onnxmltools
import onnx


def core_onnx(core_model):
    """
    Converts CoreML model into ONNX format
    Returns:
    	ONNX ML model 
    """
    c_model = str(core_model)
    coreml_model = coremltools.utils.load_spec(c_model)
    onnx_model = onnxmltools.convert_coreml(coreml_model)
    #onnxmltools.utils.save_model(onnx_model, 'model.onnx')
    return onnx_model


