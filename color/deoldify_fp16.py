import cv2
import numpy as np
import onnxruntime


class DEOLDIFY:
    def __init__(self, model_path="deoldify.onnx", device='cpu'):
        session_options = onnxruntime.SessionOptions()
        session_options.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_ENABLE_ALL
        providers = ["CPUExecutionProvider"]
        if device == 'cuda':
            providers = [("CUDAExecutionProvider", {"cudnn_conv_algo_search": "DEFAULT"}),"CPUExecutionProvider"]
        self.session = onnxruntime.InferenceSession(model_path, sess_options=session_options, providers=providers)
        self.resolution = self.session.get_inputs()[0].shape[-2:]

        
    def colorize(self, image, r_factor):
    
        # preprocess image:
        targetLAB = cv2.cvtColor(image,cv2.COLOR_BGR2LAB)
        targetL,_,_=cv2.split(targetLAB)

        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        h, w, channels = image.shape

        image = cv2.resize(image,(r_factor, r_factor))
        image = image.astype(np.float16)  
        image = image.transpose((2, 0, 1))
        image = np.expand_dims(image, axis=0).astype(np.float16)
        
        # run deoldify:
        colorized = self.session.run(None, {(self.session.get_inputs()[0].name):image})[0][0]
        
        # postprocess image:
        colorized = colorized.transpose(1,2,0).astype(np.float32)
        colorized = colorized.astype(np.uint8)
        colorized = cv2.resize(colorized,(w,h))
        colorized = cv2.GaussianBlur(colorized,(13,13),0)
        colorizedLAB = cv2.cvtColor(colorized,cv2.COLOR_RGB2LAB)
        L,A,B=cv2.split(colorizedLAB)
        colorized = cv2.merge((targetL,A,B))
        colorized = cv2.cvtColor(colorized,cv2.COLOR_LAB2BGR)
                      
        return colorized

