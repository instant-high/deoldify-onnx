import os
import cv2
import numpy as np
import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
from color.deoldify import DEOLDIFY
import onnxruntime as rt

rt.set_default_logger_severity(3)

# old model - you cannot set render_factor
#from color.deoldify_fp16 import DEOLDIFY
#colorizer = DEOLDIFY(model_path="color/deoldify_fp16.onnx", device="cpu")
#from color.deoldify import DEOLDIFY
#colorizer = DEOLDIFY(model_path="color/deoldify.onnx", device="cuda")

from color.deoldify import DEOLDIFY
colorizer = DEOLDIFY(model_path="color/ColorizeArtistic_dyn.onnx", device="cuda")

    
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
    
def process_image(file_path):
    render_factor = 8
    render_factor = render_factor * 32
    
    image = cv2.imread(file_path)
    if image is None:
        messagebox.showerror("Error", "Failed to load the image.")
        return
    
    # Resize the image if it's too big
    image = resize_image(image)
    
    colorized = colorizer.colorize(image, render_factor)
    
    #colorized = adjust_saturation(colorized, 1) # 0.1 - 2.0
  
    # Convert the OpenCV BGR image to RGB
    colorized_rgb = cv2.cvtColor(colorized, cv2.COLOR_BGR2RGB)
    
    # Convert the colorized image to PIL format
    img_colorized = Image.fromarray(colorized_rgb)
    
    # Get the screen width and height
    screen_width = root.winfo_screenwidth()
    screen_height = root.winfo_screenheight()
    
    # Resize the colorized image to fit the screen dimensions
    img_width, img_height = img_colorized.size
    aspect_ratio = img_width / img_height
    max_width = screen_width - 200  # Adjusted for padding
    max_height = screen_height - 100  # Adjusted for padding
    
    if max_width / aspect_ratio < max_height:
        new_width = max_width
        new_height = int(new_width / aspect_ratio)
    else:
        new_height = max_height
        new_width = int(new_height * aspect_ratio)
    
    img_colorized = img_colorized.resize((new_width, new_height))
    
    # Create a PhotoImage object to display in the Tkinter window
    img_colorized_tk = ImageTk.PhotoImage(img_colorized)
    
    # Update the label with the colorized image
    colorized_label.configure(image=img_colorized_tk)
    colorized_label.image = img_colorized_tk
    
    # Set the window size to fit the image and center it on the screen
    root.geometry(f"{new_width}x{new_height}+{(screen_width - new_width) // 2}+{(screen_height - new_height) // 2}")
    
    # Save the colorized image
    result_file = os.path.splitext(file_path)[0]
    result_extension = os.path.splitext(file_path)[1]
    result_image = result_file + "_colorized" + result_extension    
    img_colorized.save(result_image)
    messagebox.showinfo("Done", "File saved as " + result_image)
    
    # Ensure the window stays centered after displaying the image
    root.update_idletasks()



def main():
    global root, original_label, colorized_label
    root = tk.Tk()
    root.title("Image Colorization - deoldify.onnx")
    
    # Calculate the center position
    screen_width = root.winfo_screenwidth()
    screen_height = root.winfo_screenheight()
    window_width = 400  # Adjust as needed
    window_height = 70  # Adjust as needed
    x = (screen_width - window_width) // 2
    y = (screen_height - window_height) // 2
    
    # Set the window size and position
    root.geometry(f"{window_width}x{window_height}+{x}+{y}")
    
    select_button = tk.Button(root, text="Select Image", command=select_image)
    select_button.pack(pady=10)
    
    original_label = tk.Label(root)
    original_label.pack(side=tk.LEFT, padx=10)
    
    colorized_label = tk.Label(root)
    colorized_label.pack(side=tk.RIGHT, padx=10)
    
    root.mainloop()


if __name__ == "__main__":
    main()
