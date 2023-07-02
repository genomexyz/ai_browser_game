import onnx
import onnxruntime
import torch
import numpy as np

# Path to the ONNX model file
model_path = "model.onnx"

# Load the ONNX model
model = onnx.load(model_path)

# Print model metadata
print("ONNX Model Metadata:")
print("Model IR Version:", model.ir_version)
print("Producer Name:", model.producer_name)
print("Producer Version:", model.producer_version)
print("Model Domain:", model.domain)
print("Model Description:", model.doc_string)

# Print input information
print("\nInput Information:")
for input in model.graph.input:
    print("Name:", input.name)
    print("Type:", input.type)
    print("Shape:", input.type.tensor_type.shape)
    print()

# Print output information
print("\nOutput Information:")
for output in model.graph.output:
    print("Name:", output.name)
    print("Type:", output.type)
    print("Shape:", output.type.tensor_type.shape)
    print()

# Load the ONNX model
sess = onnxruntime.InferenceSession(model_path)

input_data = np.random.randn(1, 25**2).astype(np.float32)  # Adjust shape and data type as per your model
predictions = sess.run(['action'], {'input': input_data})
print(predictions)