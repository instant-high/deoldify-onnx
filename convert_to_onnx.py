import os

import torch

# import both to convert all models:
from deoldify.generators import gen_inference_deep
from deoldify.generators import gen_inference_wide

import torch.nn as nn
from pathlib import Path
from fastai.vision.data import normalize_funcs, imagenet_stats

norm, denorm = normalize_funcs(*imagenet_stats)

class ImageScaleInput(nn.Module):
	def __init__(self):
		super().__init__()
		self.norm = norm

	def forward(self, x):
		out = (x.div(255.0)).type(torch.float32)
		out, _ = self.norm((out, out), do_x=True)
		# out = out.unsqueeze(0)
		return out


class ImageScaleOutput(nn.Module):
	def __init__(self):
		super().__init__()
		self.denorm = denorm

	def forward(self, x):
		out = self.denorm(x, do_x=True)
		out = out.float().clamp(min=0, max=1)
		out = self.denorm(out, do_x=False)
		out = (out.mul(255.0)).type(torch.float32)
		return out


onnx_path = 'ColorizeStable_gen.onnx'

root_folder=Path('./deoldify')


# select the original model to be converted:
raw_model = gen_inference_deep(root_folder=Path('./deoldify'), weights_name='./deoldify/ColorizeArtistic_gen').model
onnx_path = 'ColorizeArtistic_gen.onnx'

#raw_model = gen_inference_wide(root_folder=Path('./deoldify'), weights_name='./deoldify/ColorizeStable_gen').model
#onnx_path = 'ColorizeStable_gen.onnx'

#raw_model = gen_inference_wide(root_folder=Path('./deoldify'), weights_name='./deoldify/DeoldifyVideo_gen').model
#onnx_path = 'DeoldifyVideo_gen.onnx'



# select input size you want to use for onnx model - dynamic seems to be not working  - 256 and 512 successfully tested:
dummy_input = torch.randn(1, 3, 256, 256)
#dummy_input = torch.randn(1, 3, 512, 512)

# if torch cuda:
dummy_input = dummy_input.to('cuda')

final_pytorch_model = nn.Sequential(ImageScaleInput(), raw_model, ImageScaleOutput())

torch.onnx.export(
	final_pytorch_model,
	dummy_input,
	onnx_path,
	do_constant_folding=False,
	input_names=['input'],
	output_names=['output'],
	opset_version=10
)