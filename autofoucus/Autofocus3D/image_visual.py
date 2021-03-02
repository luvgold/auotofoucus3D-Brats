from PIL import Image
import PIL.Image, PIL.ImageTk
import numpy as np
import matplotlib.pyplot as plt
import tkinter
from tkinter import *
from Autofocus3D.load_data import load_nii_data

def seg_visualize_2D( X, Y, bfactor, light_fac=1 ):
    cmap = plt.get_cmap('bone')
    X=X[:,:,1]
    M = cmap(X / 400)
    h, w, _ = M.shape
    for i in range(h):
        for j in range(w):
            #   ET
            if Y[i, j, 2] == 1:
                M[i, j] = (255, 0, 0, 255 * bfactor) + M[i, j, 3] * (1 - bfactor)
                continue
            #     ED
            if Y[i, j, 1] == 1:
                M[i, j] = (255, 255, 0, 255 * bfactor) + M[i, j, 3] * (1 - bfactor)
                continue
            #     NET
            if Y[i, j, 0] == 1:
                M[i, j] = (0, 255, 0, 255* bfactor) + M[i, j, 3] * (1 - bfactor)
                continue


    plt.subplot(121)
    plt.imshow(X,'bone')
    plt.subplot(122)
    plt.imshow(M*light_fac)
    plt.show()


def segAdd(X,Y,bfactor, light_fac=1):
    M=X.copy()
    h, w, _ = M.shape
    for i in range(h):
        for j in range(w):
            #   ET
            if Y[i, j, 2] == 1:
                M[i, j] = (255* bfactor, 0, 0, )  + M[i, j] * (1 - bfactor)
                continue
            #     ED
            if Y[i, j, 1] == 1:
                M[i, j] = (255* bfactor, 255* bfactor, 0) + M[i, j] * (1 - bfactor)
                continue
            #     NET
            if Y[i, j, 0] == 1:
                M[i, j] = (0, 255* bfactor, 0) + M[i, j] * (1 - bfactor)
                continue
    return M
def seg_visualize_3D(X,Y,bfactor,light_fac=1,**kwargs):
    d, h, w, _ = X.shape
    window = Tk()
    window.attributes("-alpha", 1)
    window.geometry('800x500')
    window.title("CT序列")
    value = IntVar()

    def show(text):  # 注意，Scale的回调函数需要给定形参，当触发时会将Scale的值传给函数
        slice = int(text)
        interval = 200
        cmap = plt.get_cmap('bone')
        img = X[ slice, :, :, 1]
        img = cmap(img / 400)
        rgb_img = np.delete(img, 3, 2)
        rgb_img = rgb_img * 255

        img = rgb_img.astype("uint8")
        seg = segAdd(img, Y[slice], bfactor,light_fac)
        seg = seg.astype("uint8")

        img = PIL.Image.fromarray(img)
        seg = PIL.Image.fromarray(seg)
        if kwargs.get("resize") == True or kwargs.get("reshape"):
            if kwargs.get("reshape"):
                img = img.resize(kwargs["reshape"])
                seg = seg.resize(kwargs["reshape"])
                interval=kwargs["reshape"][1]
            else:

                img = img.resize((200,200))
                seg = seg.resize((200,200))
        global tk_img, tk_seg
        tk_img = PIL.ImageTk.PhotoImage(img)
        tk_seg = PIL.ImageTk.PhotoImage(seg)
        margin = window.winfo_width()//2-interval
        canvas.create_image(margin, 0, image=tk_img, anchor="nw")
        canvas.create_image(margin+interval, 0, image=tk_seg, anchor="nw")

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
    canvas = tkinter.Canvas(frame, bd=0,width=800,height=800)
    canvas.grid(row=0, column=0, sticky=tkinter.N + tkinter.S + tkinter.E + tkinter.W)
    canvas.config(scrollregion=canvas.bbox(tkinter.ALL))
    frame.pack()

    window.mainloop()

#     2D
# X, Y = load_nii_data("./Dataset/LGG", "seg",1, (22, 200, 200))
# seg_visualize_2D(X[0,10],Y[0,10],0.8)

#     3D
X, Y = load_nii_data("./Dataset/LGG", "seg",1, (120,200,200))
seg_visualize_3D(X[0],Y[0],0.8)