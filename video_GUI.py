import os
import cv2
import numpy as np
import subprocess
import platform
import tkinter as tk
from tkinter import filedialog, messagebox
from argparse import Namespace
from tqdm import tqdm
import onnxruntime as rt
rt.set_default_logger_severity(3)

# old model - you cannot set render_factor
#from color.deoldify_fp16 import DEOLDIFY
#colorizer = DEOLDIFY(model_path="color/deoldify_fp16.onnx", device="cpu")
#from color.deoldify import DEOLDIFY
#colorizer = DEOLDIFY(model_path="color/deoldify.onnx", device="cuda")

from color.deoldify import DEOLDIFY
colorizer = DEOLDIFY(model_path="color/ColorizeArtistic_dyn.onnx", device="cuda")


class DEOLDIFY_GUI:
    def __init__(self, master):
        self.master = master
        master.title("Video Colorizer - Deoldify-ONNX")

        # Calculate the center position
        screen_width = master.winfo_screenwidth()
        screen_height = master.winfo_screenheight()
        window_width = 440  # Adjust as needed
        window_height = 120  # Adjust as needed
        x = (screen_width - window_width) // 2
        y = (screen_height - window_height) // 2
    
        # Set the window size and position
        master.geometry(f"{window_width}x{window_height}+{x}+{y}")
    
        self.source_label = tk.Label(master, text="Source Video:")
        self.source_label.grid(row=0, column=0, sticky='w')

        self.source_path = tk.StringVar()
        self.source_entry = tk.Entry(master, textvariable=self.source_path, width=50)
        self.source_entry.grid(row=0, column=1)

        self.source_button = tk.Button(master, text="Browse", command=self.browse_source)
        self.source_button.grid(row=0, column=2)

        self.result_label = tk.Label(master, text="Result Video:")
        self.result_label.grid(row=1, column=0, sticky='w')

        self.result_path = tk.StringVar()
        self.result_entry = tk.Entry(master, textvariable=self.result_path, width=50)
        self.result_entry.grid(row=1, column=1)

        self.result_button = tk.Button(master, text="Browse", command=self.browse_result)
        self.result_button.grid(row=1, column=2)

        self.audio_var = tk.BooleanVar()
        self.audio_checkbox = tk.Checkbutton(master, text="Keep Audio", variable=self.audio_var)
        self.audio_checkbox.grid(row=2, column=0, columnspan=3)

        self.run_button = tk.Button(master, text="Run", command=self.run_colorizer)
        self.run_button.grid(row=3, columnspan=3)

    def browse_source(self):
        file_path = filedialog.askopenfilename(filetypes=[("Video files", "*.mp4;*.avi;*.mkv")])
        self.source_path.set(file_path)

    def browse_result(self):
        file_path = filedialog.asksaveasfilename(defaultextension=".mp4", filetypes=[("Video files", "*.mp4")])
        self.result_path.set(file_path)

    def run_colorizer(self):
        source = self.source_path.get()
        result = self.result_path.get()
        if not source or not result:
            messagebox.showerror("Error", "Please select source and result paths.")
            return

        opt = Namespace(source=source, result=result, audio=self.audio_var.get())
        self.colorize_video(opt)
        messagebox.showinfo("Done", f"Inference done. Output file: {result}")
        
    
    def colorize_video(self, opt):
        render_factor = 8
        render_factor = render_factor * 32
    
        video = cv2.VideoCapture(opt.source)
        w = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
        n_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = video.get(cv2.CAP_PROP_FPS)

        if opt.audio:
            writer = cv2.VideoWriter('temp.mp4', cv2.VideoWriter_fourcc('m', 'p', '4', 'v'), fps, (w, h))
        else:
            writer = cv2.VideoWriter(opt.result, cv2.VideoWriter_fourcc('m', 'p', '4', 'v'), fps, (w, h))

        for frame_idx in tqdm(range(n_frames)):
            ret, frame = video.read()
            if not ret:
                break
            result = colorizer.colorize(frame, render_factor)
            
            #result = adjust_saturation(result, 1) # 0.1 - 2.0
            
            writer.write(result)

            cv2.imshow ("Result - press ESC to stop",result)
            k = cv2.waitKey(1)
            if k == 27:
                writer.release()
                break
        
        writer.release()
        video.release()
        cv2.destroyAllWindows()
        if opt.audio:
            # lossless remuxing audio/video
            command = 'ffmpeg.exe -y -vn -i ' + '"' + opt.source + '"' + ' -an -i ' + 'temp.mp4' + \
                      ' -c:v copy -acodec libmp3lame -ac 2 -ar 44100 -ab 128000 -map 0:1 -map 1:0 -shortest ' + '"' + opt.result + '"'
            subprocess.call(command, shell=platform.system() != 'Windows')          
            os.remove('temp.mp4')

def adjust_saturation(image, saturation_factor):
    # Convert BGR image to HSV
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    # Split into Hue, Saturation, and Value channels
    h, s, v = cv2.split(hsv)
    # Scale the saturation channel
    s = np.clip(s * saturation_factor, 0, 255).astype(np.uint8)
    # Merge the channels back
    adjusted_hsv = cv2.merge([h, s, v])
    # Convert back to BGR
    adjusted_bgr = cv2.cvtColor(adjusted_hsv, cv2.COLOR_HSV2BGR)
    return adjusted_bgr
    
def main():

    root = tk.Tk()
    deoldify_gui = DEOLDIFY_GUI(root)
    root.mainloop()

if __name__ == "__main__":
    main()
