# deoldify-onnx

Simple image and video colorization using onnx converted deoldify model.

Easy to install. Can be run on CPU or nVidia GPU

ffmpeg for video colorzation required.

Added floating point 16 model for 100% faster inference and simple GUI version.

For inference run:

Image:
python image.py --source_image "image.jpg"

Video:
python video.py --source "video.mp4" --result "video_colorized.mp4" --audio

Image example:
![colorizer1](https://github.com/instant-high/deoldify-onnx/assets/77229558/171642dd-9034-4ca7-8d29-c07c6e5e9f0a)


https://github.com/instant-high/deoldify-onnx/assets/77229558/3824e96d-fffc-494e-8ce1-193e6a77c8b6

https://github.com/instant-high/deoldify-onnx/assets/77229558/543e1dd1-27da-4c63-95a9-9c0696adea51

.

link to download the converted deoldify.onnx from this repository:

https://github.com/KeepGoing2019HaHa/AI-application/tree/master/deoldify/ncnn
