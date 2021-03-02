import os
import SimpleITK as sitk
import numpy as np
from scipy import ndimage


def load_nii_data(train_path,mask_name,train_num,re_shape=(1,1,1)):
    masks=[]
    images=[]
    d,h,w=re_shape
    fileTrainCTList = os.listdir(train_path)
    # fileTestCTList = os.listdir(testCTPath)
    patient_num = int(fileTrainCTList.__len__() )

    for n in range(train_num):
        print("N_Patient: "+str(n))
        file=fileTrainCTList[n]
        datas=os.listdir(train_path+'/'+file)
        image=[]
        image_input=datas.__len__()
        for i in range(image_input):
            if mask_name in datas[i]:
                mask = sitk.ReadImage(train_path+'/'+file+'/'+datas[i])
                re_mask = resample(mask,d, h, w)
                tem = np.expand_dims(re_mask, -1)
                NET = np.array(tem[:] == 1).astype(int)
                ED = np.array(tem[:] == 2).astype(int)
                ET = np.array(tem[:] == 4).astype(int)
                mask = np.concatenate((NET, ED, ET), axis=-1)
                masks.append(mask)
                continue
            elif image==[]:
                image=sitk.ReadImage(train_path+'/'+file+'/'+datas[i])
                normalize(image)
                re_image = resample(image, d,h,w)
                image=np.expand_dims(re_image,-1)
            else:
                im = sitk.ReadImage(train_path+'/'+file+'/'+datas[i])
                normalize(im)
                re_im=resample(im, d, h, w)
                im=np.expand_dims(re_im,-1)
                image =  np.concatenate((image,im), axis =-1)
            if i==image_input-1:
                images.append(image)
    return np.array(images, 'float32'),np.array(masks, 'float32')

def resample(image,w,h,d):
    # 如果shape小于目标shape则进行升采样,若大于则降采样
    new_shape=(w,h,d)
    data=np.array(sitk.GetArrayFromImage(image))
    if w == h == d == 1:
        return data
    w1, h1, d1 = data.shape
    change_rate = (w1/w,h1/h,d1/d)
    new_spacing = change_rate
    inputSpacing = image.GetSpacing()
    resize_factor = (inputSpacing[0]/ new_spacing[0],inputSpacing[1]/ new_spacing[1],inputSpacing[2]/ new_spacing[2])
    # 得到refactor即可获得新的shape
    new_real_shape = (data.shape[0] * resize_factor[0],data.shape[1] * resize_factor[1],data.shape[2] * resize_factor[2])
    # 取整
    new_shape = np.round(new_real_shape)
    # 取整后获得的整数shape再次计算出一个refactor,便于求整
    refactor = new_shape / data.shape
    # 这里用了样条插值的方法,默认order=3
    new_volume = ndimage.zoom(data, zoom=refactor)
    return new_volume

def normalize(img):
    WINDOW_LEVEL = (1000, -500)
    img_norm = sitk.Cast(sitk.IntensityWindowing(img,
                                                 windowMinimum=WINDOW_LEVEL[1] - WINDOW_LEVEL[0] / 2.0,
                                                 windowMaximum=WINDOW_LEVEL[1] + WINDOW_LEVEL[0] / 2.0),
                         sitk.sitkUInt8)
    img_norm_arr = sitk.GetArrayFromImage(img_norm)
    return img_norm_arr

# 此网络中并没采用
def tem_normalize(arr):
    """
    Linear normalization
    http://en.wikipedia.org/wiki/Normalization_%28image_processing%29
    """
    arr = arr.astype('float32')
    # Do not touch the alpha channel
    for i in range(3):
        minval = arr[i, ...].min()
        maxval = arr[i, ...].max()
        if minval != maxval:
            arr[i, ...] -= minval
            arr[i, ...] *= (1024.0 / (maxval - minval))
    return arr

# images,masks=load_nii_data("./Dataset/LGG","seg",20)
# print(images.shape)
# print(masks.shape)
# image = sitk.ReadImage("./Dataset/LGG/Brats18_2013_0_1/Brats18_2013_0_1_flair.nii.gz")
#
# re_image=resample(image,64,64,64)
# print(re_image.shape)