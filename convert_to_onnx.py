'''
run this script in the original deoldify repo
'''

import os
import torch
from deoldify.generators import gen_inference_deep
from deoldify.generators import gen_inference_wide
import torch.nn as nn
from pathlib import Path

from fastai.vision.data import normalize_funcs, imagenet_stats

norm, denorm = normalize_funcs(*imagenet_stats)

class ImageScaleInput(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        out = (x.div(255.0)).type(torch.float32)
        out, _ = norm((out, out))
        return out

class ImageScaleOutput(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        out = denorm(x)
        out = out.float().clamp(min=0, max=1)
        out = (out.mul(255.0)).type(torch.float32)
        return out

root_folder = Path('./deoldify')

# select the original model to be converted:
#raw_model = gen_inference_deep(root_folder=Path('./deoldify'), weights_name='./deoldify/ColorizeArtistic_gen').model
#onnx_path = 'ColorizeArtistic_dyn.onnx'

#raw_model = gen_inference_wide(root_folder=Path('./deoldify'), weights_name='./deoldify/ColorizeStable_gen').model
#onnx_path = 'ColorizeStable_dyn.onnx'

raw_model = gen_inference_wide(root_folder=Path('./deoldify'), weights_name='./deoldify/DeoldifyVideo_gen').model
onnx_path = 'DeoldifyVideo_dyn.onnx'

dummy_input = torch.randn(1, 3, 256, 256)

# Wenn CUDA verfügbar ist, auf CUDA umschalten
dummy_input = dummy_input.to('cuda')

final_pytorch_model = nn.Sequential(ImageScaleInput(), raw_model, ImageScaleOutput())

torch.onnx.export(
    final_pytorch_model,
    dummy_input,
    onnx_path,
    do_constant_folding=False,
    input_names=['input'],
    output_names=['output'],
    opset_version=12,
    dynamic_axes={'input': {2: 'height', 3: 'width'}, 'output': {2: 'height', 3: 'width'}}
)
