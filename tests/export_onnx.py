import numpy as np
import time
import onnx
import onnxruntime
import torch.onnx
from batch_face import LandmarkPredictor

backbone = "PFLD"

predictor = LandmarkPredictor(-1, backbone=backbone)

# Create the super-resolution model by using the above model definition.
torch_model = predictor.model

batch_size = 1  # just a random number

# set the model to inference mode
torch_model.eval()

x = torch.randn(batch_size, 3, 112, 112, requires_grad=True)
start = time.time()
for i in range(100):
    torch_out = torch_model(x)
print(time.time() - start)

file = backbone + ".onnx"

# Export the model
torch.onnx.export(
    torch_model,  # model being run
    x,  # model input (or a tuple for multiple inputs)
    file,  # where to save the model (can be a file or file-like object)
    export_params=True,  # store the trained parameter weights inside the model file
    opset_version=10,  # the ONNX version to export the model to
    do_constant_folding=True,  # whether to execute constant folding for optimization
    input_names=["input"],  # the model's input names
    output_names=["output"],  # the model's output names
    dynamic_axes={
        "input": {0: "batch_size"},  # variable lenght axes
        "output": {0: "batch_size"},
    },
)

onnx_model = onnx.load(file)
onnx.checker.check_model(onnx_model)


ort_session = onnxruntime.InferenceSession(file)


def to_numpy(tensor):
    return (
        tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()
    )


# compute ONNX Runtime output prediction
ort_inputs = {ort_session.get_inputs()[0].name: to_numpy(x)}

start = time.time()
for i in range(100):
    ort_outs = ort_session.run(None, ort_inputs)
print(time.time() - start)
print(ort_outs[0].shape)

# compare ONNX Runtime and PyTorch results
np.testing.assert_allclose(to_numpy(torch_out), ort_outs[0], rtol=1e-03, atol=1e-05)

print("Exported model has been tested with ONNXRuntime, and the result looks good!")
