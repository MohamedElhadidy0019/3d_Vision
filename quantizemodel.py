import onnx
from onnxruntime.quantization import quantize_dynamic, QuantType

# Load the original ONNX model
model_path = "best_infer.onnx"
model = onnx.load(model_path)

# Quantize the model
quantized_model_path = "model_quantized_infer.onnx"
quantized_model = quantize_dynamic(
    model_path, 
    quantized_model_path, 
    weight_type=QuantType.QUInt8,  # Use QuantType.QUInt8 if neededm
    # nodes_to_exclude=['/model.0/conv/Conv_quant']
)
# quantize_dynamic(input_model, output_model, weight_type=QuantType.QInt8, nodes_to_exclude=['/conv1/Conv'])


# Verify the quantized model
quantized_model = onnx.load(quantized_model_path)
onnx.checker.check_model(quantized_model)
print("Quantized model is valid and saved at:", quantized_model_path)

# python -m onnxruntime.quantization.preprocess --input yolov8x-seg_float.onnx --output yolov8x-seg_float_infer.onnx
# python -m onnxruntime.quantization.preprocess --input best.onnx --output best_infer.onnx
