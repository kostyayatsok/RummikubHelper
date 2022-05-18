import tf2onnx
import tensorflow as tf
import onnxruntime as rt
import numpy as np

x = np.random.randn(1, 48, 48, 3).astype(np.float32)

FILTER_NAME = 'rummi_936.h5'
model = tf.keras.models.load_model(FILTER_NAME)
preds = model(x)

spec = (tf.TensorSpec((None, 48, 48, 3), tf.float32, name="input_1"),)
output_path = model.name + ".onnx"
print(output_path)
model_proto, _ = tf2onnx.convert.from_keras(model, input_signature=spec, opset=13, output_path=output_path)
output_names = [n.name for n in model_proto.graph.output]
print(output_names)

providers = ['CPUExecutionProvider']
m = rt.InferenceSession(output_path, providers=providers)
onnx_pred = m.run(["tf.reshape_2"], {"input_1": x})

print('ONNX Predicted:', onnx_pred[0])

# make sure ONNX and keras have the same results
np.testing.assert_allclose(preds, onnx_pred[0], rtol=1e-5)
