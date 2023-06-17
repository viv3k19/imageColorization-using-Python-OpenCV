
import tkinter as tk
from tkinter import *
from tkinter import filedialog
from PIL import Image, ImageTk
import os
import numpy as np
import cv2 as cv
import os.path
import matplotlib
matplotlib.use('Agg')

import sys
import os

if os.environ.get('DISPLAY','') == '':
    print('no display found. Using :0.0')
    os.environ.__setitem__('DISPLAY', ':0.0')

numpy_file = np.load('./pts_in_hull.npy')
Caffe_net = cv.dnn.readNetFromCaffe("./models/colorization_deploy_v2.prototxt", "./models/colorization_release_v2.caffemodel")
numpy_file = numpy_file.transpose().reshape(2, 313, 1, 1)

class Window(Frame):
    def __init__(self, master=None):
        Frame.__init__(self, master)

        self.master = master
        self.pos = []
        self.master.title("B&W Image Colorization")
        self.pack(fill=BOTH, expand=1)

        self.canvas_frame = Frame(self)
        self.canvas_frame.pack(side=TOP, padx=10, pady=10)

        self.original_canvas = tk.Canvas(self.canvas_frame)
        self.original_canvas.pack(side=LEFT, fill=tk.BOTH, expand=True)

        self.colorized_canvas = tk.Canvas(self.canvas_frame)
        self.colorized_canvas.pack(side=LEFT, fill=tk.BOTH, expand=True)

        self.button_frame = Frame(self)
        self.button_frame.pack(side=BOTTOM, padx=10, pady=10)

        self.upload_button = Button(self.button_frame, text="Upload Image", command=self.uploadImage)
        self.upload_button.pack(side=LEFT, padx=10, pady=10)

        self.color_button = Button(self.button_frame, text="Generate Color Image", command=self.color)
        self.color_button.pack(side=LEFT, padx=10, pady=10)

        self.original_image = None
        self.colorized_image = None

    def uploadImage(self):
        filename = filedialog.askopenfilename(initialdir=os.getcwd())
        if not filename:
            return
        load = Image.open(filename)

        load = load.resize((480, 360), Image.ANTIALIAS)

        if self.original_image is None:
            w, h = load.size
            self.original_render = ImageTk.PhotoImage(load)
            self.original_image = self.original_canvas.create_image((w / 2, h / 2), image=self.original_render)
        else:
            self.original_canvas.delete(self.original_image)
            w, h = load.size
            self.original_render = ImageTk.PhotoImage(load)
            self.original_image = self.original_canvas.create_image((w / 2, h / 2), image=self.original_render)

        frame = cv.imread(filename)

        Caffe_net.getLayer(Caffe_net.getLayerId('class8_ab')).blobs = [numpy_file.astype(np.float32)]
        Caffe_net.getLayer(Caffe_net.getLayerId('conv8_313_rh')).blobs = [np.full([1, 313], 2.606, np.float32)]

        input_width = 224
        input_height = 224

        rgb_img = (frame[:,:,[2, 1, 0]] * 1.0 / 255).astype(np.float32)
        lab_img = cv.cvtColor(rgb_img, cv.COLOR_RGB2Lab)
        l_channel = lab_img[:,:,0]

        l_channel_resize = cv.resize(l_channel, (input_width, input_height))
        l_channel_resize -= 50

        Caffe_net.setInput(cv.dnn.blobFromImage(l_channel_resize))
        ab_channel = Caffe_net.forward()[0,:,:,:].transpose((1,2,0))

        (original_height,original_width) = rgb_img.shape[:2]
        ab_channel_us = cv.resize(ab_channel, (original_width, original_height))
        lab_output = np.concatenate((l_channel[:,:,np.newaxis],ab_channel_us),axis=2)
        bgr_output = np.clip(cv.cvtColor(lab_output, cv.COLOR_Lab2BGR), 0, 1)

        cv.imwrite("./result.png", (bgr_output*255).astype(np.uint8))

    def color(self):
        load = Image.open("./result.png")
        load = load.resize((480, 360), Image.ANTIALIAS)

        if self.colorized_image is None:
            w, h = load.size
            self.colorized_render = ImageTk.PhotoImage(load)
            self.colorized_image = self.colorized_canvas.create_image((w / 2, h / 2), image=self.colorized_render)
        else:
            self.colorized_canvas.delete(self.colorized_image)
            w, h = load.size
            self.colorized_render = ImageTk.PhotoImage(load)
            self.colorized_image = self.colorized_canvas.create_image((w / 2, h / 2), image=self.colorized_render)

root = tk.Tk()
root.geometry("1000x400")
root.title("B&W Image Colorization GUI")

app = Window(root)
app.pack(fill=tk.BOTH, expand=1)
root.mainloop()
