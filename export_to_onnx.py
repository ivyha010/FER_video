import torch
from transformers import AutoModelForImageClassification

model = AutoModelForImageClassification.from_pretrained("SeanHowFun/face-emotion-detection")
dummy_input = torch.randn(1, 3, 224, 224)

torch.onnx.export(
    model,
    dummy_input,
    "fer2013_vit.onnx",
    input_names=["input"],
    output_names=["output"],
    dynamic_axes={"input": {0: "batch"}, "output": {0: "batch"}},
    opset_version=18
)