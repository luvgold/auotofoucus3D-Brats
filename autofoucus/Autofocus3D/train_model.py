import matplotlib.pyplot as plt
from Autofocus3D.Model import build_model,build_graph
from Autofocus3D.load_data import load_nii_data,normalize,resample
from keras.callbacks import ModelCheckpoint
import numpy as np
if __name__ == '__main__':
##############  mhd文件导入 ############################
    # filterSize = np.array([[3, 3, 3], [3, 3, 3]])
    # padSize = np.round(filterSize / 2).astype(int)
    #
    # train_CT_path = "Dataset/MICCAI/Train_Resamp/CT/"
    # train_MASK_path = "Dataset/MICCAI/Train_Resamp/Mask/"
    # X, Y = shared_dataset(train_CT_path, train_MASK_path, padSize)
    # input_size=X.shape
    # output_size=Y.shape
    # X=X.reshape(20,1,input_size[1],input_size[2],input_size[3])
    # Y=Y.reshape(20, 1, output_size[1], output_size[2], output_size[3])
    # print(X.shape)
##########################################################
    batch=10
    X,Y=load_nii_data("./Dataset/LGG","seg",batch,(22,22,22))
    plt.imshow(X[0,15,:,:,0],cmap='bone')
    plt.show()
#########  shape: num,c,d,h,w ##########
    # X.transpose(0, -1, 1, 2, 3)
    # Y.transpose(0, -1, 1, 2, 3)
    model=build_model()

    model.compile(optimizer='adam', loss='mean_squared_error', metrics=['accuracy'],lr=0.0001)
    batch=X.shape[0]
    ##########################################################################################
    checkpointer = ModelCheckpoint('./WeightSave/model.{epoch:03d}.hdf5', period=100)

    model.fit(X, Y, batch_size=batch, epochs=10, callbacks=[checkpointer], validation_split=0.2)
    print(X.shape)
    print(Y.shape)
    model.summary()
    build_graph(model)