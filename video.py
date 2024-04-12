
import os
import sys
import cv2
import numpy as np
import subprocess
import platform

from argparse import ArgumentParser
from tqdm import tqdm

import onnxruntime as rt
rt.set_default_logger_severity(3)

parser = ArgumentParser()
parser.add_argument("--source", help="path to source video")
parser.add_argument("--result", help="path to result video")
parser.add_argument("--audio", default=False, action="store_true", help="Keep audio")
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


video = cv2.VideoCapture(opt.source)

w = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
h = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
n_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))    
fps = video.get(cv2.CAP_PROP_FPS)

if opt.audio:
    writer = cv2.VideoWriter('temp.mp4',cv2.VideoWriter_fourcc('m','p','4','v'), fps, (w, h))
else:
    writer = cv2.VideoWriter(opt.result,cv2.VideoWriter_fourcc('m','p','4','v'), fps, (w, h))

for frame_idx in tqdm(range(n_frames)):

    ret, frame = video.read()
    if not ret:
        break

    result = colorizer.colorize(frame, render_factor)
    
    writer.write(result)
    cv2.imshow ("Result",result)
    k = cv2.waitKey(1)
    if k == 27:
        writer.release()
        break

if opt.audio:
    # lossless remuxing audio/video
    command = 'ffmpeg.exe -y -vn -i ' + '"' + opt.source + '"' + ' -an -i ' + 'temp.mp4' + ' -c:v copy -acodec libmp3lame -ac 2 -ar 44100 -ab 128000 -map 0:1 -map 1:0 -shortest ' + '"' + opt.result + '"'
    subprocess.call(command, shell=platform.system() != 'Windows')
    os.remove('temp.mp4')
    