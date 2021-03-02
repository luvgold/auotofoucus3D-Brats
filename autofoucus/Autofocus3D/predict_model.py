import matplotlib.pyplot as plt
from Autofocus3D.Model import build_model
from Autofocus3D.load_data import load_nii_data,normalize,resample
from keras.callbacks import ModelCheckpoint
import numpy as np
from Autofocus3D.image_visual import seg_visualize_3D

X,Y=load_nii_data("./Dataset/test","seg",5,(22,22,22))
S = X.copy()
model=build_model()
model.build((None,22, 22,22,4))
model.load_weights("./WeightSave/model.1000.hdf5")

Y_T=model.predict(S)
print(Y_T[0])
Y_T=np.around(Y_T)
print(Y_T.shape)
print(Y_T.max())

seg_visualize_3D(X[3],Y_T[3],0.8,resize=True)
