import os
import cv2
import numpy as np
import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
from color.deoldify import DEOLDIFY
import onnxruntime as rt

rt.set_default_logger_severity(3)
colorizer = DEOLDIFY(model_path="color/deoldify.onnx", device="cpu")

def resize_image(image):
    max_height = root.winfo_screenheight() - 100  # Adjusted for padding
    max_width = root.winfo_screenwidth() - 200   # Adjusted for padding
    height, width = image.shape[:2]
    if height > max_height or width > max_width:
        scale = min(max_height/height, max_width/width)
        new_width = int(width * scale)
        new_height = int(height * scale)
        return cv2.resize(image, (new_width, new_height))
    return image

def select_image():
    file_path = filedialog.askopenfilename(filetypes=[("Image files", "*.jpg;*.jpeg;*.png;*.bmp")])
    if file_path:
        process_image(file_path)

def process_image(file_path):
    image = cv2.imread(file_path)
    if image is None:
        messagebox.showerror("Error", "Failed to load the image.")
        return
    
    # Resize the image if it's too big
    image = resize_image(image)
    
    colorized = colorizer.colorize(image)
    
    # Convert the OpenCV BGR image to RGB
    colorized_rgb = cv2.cvtColor(colorized, cv2.COLOR_BGR2RGB)
    
    # Convert both the original and colorized images to PIL format
    img_original = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    img_colorized = Image.fromarray(colorized_rgb)
    
    # Create PhotoImage objects to display in the Tkinter window
    img_original_tk = ImageTk.PhotoImage(img_original)
    img_colorized_tk = ImageTk.PhotoImage(img_colorized)
    
    # Update the labels with the selected image and its colorized result
    original_label.configure(image=img_original_tk)
    original_label.image = img_original_tk
    colorized_label.configure(image=img_colorized_tk)
    colorized_label.image = img_colorized_tk
    
    #
    result_file= os.path.splitext(file_path)[0]
    result_extension = os.path.splitext(file_path)[1]
    result_image = result_file + "_colorized" + result_extension    
    cv2.imwrite(result_image, colorized)
    #root.title("Image Colorization - saved as " + result_image)
    messagebox.showinfo("Done", "File saved as " + result_image)

def main():
    global root, original_label, colorized_label
    root = tk.Tk()
    root.title("Image Colorization - deoldify.onnx")
    
    select_button = tk.Button(root, text="Select Image", command=select_image)
    select_button.pack(pady=10)
    
    original_label = tk.Label(root)
    original_label.pack(side=tk.LEFT, padx=10)
    
    colorized_label = tk.Label(root)
    colorized_label.pack(side=tk.RIGHT, padx=10)
    
    root.mainloop()

if __name__ == "__main__":
    main()
