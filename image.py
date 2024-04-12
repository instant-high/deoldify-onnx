import os
import cv2
import numpy as np
from argparse import ArgumentParser

import onnxruntime as rt
rt.set_default_logger_severity(3)
parser = ArgumentParser()
parser.add_argument("--source_image", default='source.jpg', help="path to source image")
parser.add_argument("--result_image", default='result.jpg', help="path to result image")
parser.add_argument("--render_factor", type=int, default=8, help=" - ")
opt = parser.parse_args()

'''
The render factor determines the resolution at which the image is rendered for inference.
When set at a low value, the process is faster and the colors tend to be more vibrant
but the results are less stable.
original torch model accepts input divisible by 16
ONNX models currently accept only divisible by 32  
'''

#
render_factor = opt.render_factor * 32
#

# old model - you cannot set render_factor
#from color.deoldify_fp16 import DEOLDIFY
#colorizer = DEOLDIFY(model_path="color/deoldify_fp16.onnx", device="cpu")
#from color.deoldify import DEOLDIFY
#colorizer = DEOLDIFY(model_path="color/deoldify.onnx", device="cuda")

# new onnx models - render_factor - dynamic axes input:
from color.deoldify_fp16 import DEOLDIFY
colorizer = DEOLDIFY(model_path="color/ColorizeArtistic_dyn_fp16.onnx", device="cuda")
#from color.deoldify import DEOLDIFY
#colorizer = DEOLDIFY(model_path="color/ColorizeArtistic_dyn.onnx", device="cuda")

image = cv2.imread(opt.source_image)

colorized = colorizer.colorize(image, render_factor)

cv2.imwrite(opt.result_image, colorized) 
cv2.imshow("Colorized image saved - press any key",colorized)
cv2.waitKey() 
