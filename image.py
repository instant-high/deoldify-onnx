import os
import cv2
import numpy as np
from argparse import ArgumentParser

import onnxruntime as rt
rt.set_default_logger_severity(3)

from color.deoldify import DEOLDIFY
colorizer = DEOLDIFY(model_path="color/deoldify.onnx", device="cpu")

parser = ArgumentParser()
parser.add_argument("--source_image", default='source.jpg', help="path to source image")
#parser.add_argument("--result_image", default='result.jpg', help="path to result image")
opt = parser.parse_args()

image = cv2.imread(opt.source_image)

result_file= os.path.splitext(opt.source_image)[0]
result_extension = os.path.splitext(opt.source_image)[1]
opt.result_image = result_file + "_colorized" + result_extension

colorized = colorizer.colorize(image)

cv2.imwrite(opt.result_image, colorized) 
cv2.imshow("Colorized image saved - press any key",colorized)
cv2.waitKey() 
