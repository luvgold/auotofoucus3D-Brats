from Autofocus3D.load_data import load_nii_data
from Autofocus3D.image_visual import segAdd
import tkinter
import PIL.Image, PIL.ImageTk
import numpy as np
import matplotlib.pyplot as plt
from tkinter import *

def seg_visualize_3D(X,Y,bfactor,light_fac=1):
    d, h, w, _ = X.shape
    window = Tk()
    window.attributes("-alpha", 1)
    window.geometry('500x300')
    window.title("CT序列")
    value = IntVar()

    def show(text):  # 注意，Scale的回调函数需要给定形参，当触发时会将Scale的值传给函数
        slice = int(text)
        cmap = plt.get_cmap('bone')
        img = X[ slice, :, :, 1]
        img = cmap(img / 400)
        rgb_img = np.delete(img, 3, 2)
        rgb_img = rgb_img * 255
        img = rgb_img.astype("uint8")
        seg = segAdd(img, Y[slice], bfactor,light_fac)
        seg = seg.astype("uint8")
        print(slice)
        img = PIL.Image.fromarray(img)
        seg = PIL.Image.fromarray(seg)
        global tk_img, tk_seg
        tk_img = PIL.ImageTk.PhotoImage(img)
        tk_seg = PIL.ImageTk.PhotoImage(seg)
        canvas.create_image(0, 0, image=tk_img, anchor="nw")
        canvas.create_image(200, 0, image=tk_seg, anchor="nw")

    tk_img = None
    tk_seg = None
    s1 = Scale(window,
               from_=0, to=d - 1, length=300,
               resolution=1,
               showvalue=1,
               orient=HORIZONTAL,
               variable=value,
               command=show
               )
    s1.pack()

    frame = tkinter.Frame(window, bd=2)  # relief=SUNKEN)
    canvas = tkinter.Canvas(frame, bd=0)
    canvas.grid(row=0, column=0, sticky=tkinter.N + tkinter.S + tkinter.E + tkinter.W)
    canvas.config(scrollregion=canvas.bbox(tkinter.ALL))
    frame.pack()

    window.mainloop()

X, Y = load_nii_data("./Dataset/LGG", "seg",1, (22, 200, 200))
seg_visualize_3D(X[0],Y[0],0.8)
