
# coding: utf-8

# Import


from keras.layers import Conv3D, Input, RepeatVector, merge, Activation, UpSampling3D, Conv3DTranspose
from keras.layers.merge import add, concatenate
from keras.models import Model
from keras.layers.advanced_activations import PReLU, Softmax
from keras.activations import softmax
from keras import activations, initializers, regularizers
from keras.optimizers import SGD
from keras.utils import Sequence
from keras.callbacks import LearningRateScheduler
import tensorflow as tf
import keras.backend as K

import SimpleITK as sitk
import numpy as np
import matplotlib.pyplot as plt
import os
import math

import pickle
from pathlib import Path


# Network architecture

input_layer = Input(shape=(128, 128, 64, 1), name="data")

#According to table 1 from v-net paper
#encoding part
#left layer-1
conv1 = Conv3D(16, (5, 5, 5), padding='same', data_format='channels_last')(input_layer)
concatinput_1 = concatenate([input_layer]*16)
summed_1 = add([conv1, concatinput_1])
activ_1 = PReLU()(summed_1)
stride_1 = Conv3D(32, (2, 2, 2), strides=(2, 2, 2))(activ_1)
#it is recommended to do advanced activations using other layers
activ_11 = PReLU()(stride_1) 

#left_layer-2 32 channels
conv2 = Conv3D(32, (5, 5, 5), padding='same', data_format="channels_last")(activ_11)
activ_2 = PReLU()(conv2)
conv22 = Conv3D(32, (5, 5, 5), padding='same', data_format='channels_last')(activ_2)
activ_22 = PReLU()(conv22)


summed_2 = add([activ_22, activ_11])

stride_2 = Conv3D(64, (2, 2, 2), strides=(2, 2, 2))(summed_2)
activ_23 = PReLU()(stride_2)



#left_layer-3 64 channels
conv3 = Conv3D(64, (5, 5, 5), padding='same', data_format="channels_last")(activ_23)
activ_3 = PReLU()(conv3)
conv32 = Conv3D(64, (5, 5, 5), padding='same', data_format='channels_last')(activ_3)
activ_32 = PReLU()(conv32)
conv33 = Conv3D(64, (5, 5, 5), padding='same', data_format='channels_last')(activ_32)
activ_33  = PReLU()(conv33)

summed_3 = add([activ_33, activ_23])

stride_3 = Conv3D(128, (2, 2, 2), strides=(2, 2, 2))(summed_3)
activ_34 = PReLU()(stride_3)


#left_layer-4 128 channels
conv4 = Conv3D(128, (5, 5, 5), padding='same', data_format="channels_last")(activ_34)
activ_4 = PReLU()(conv4)
conv42 = Conv3D(128, (5, 5, 5), padding='same', data_format='channels_last')(activ_4)
activ_42 = PReLU()(conv42)
conv43 = Conv3D(128, (5, 5, 5), padding='same', data_format='channels_last')(activ_42)
activ_43  = PReLU()(conv43)

summed_4 = add([activ_43, activ_34])

stride_4 = Conv3D(256, (2, 2, 2), strides=(2, 2, 2))(summed_4)
activ_44 = PReLU()(stride_4)

#left_layer-5 256 channels
conv5 = Conv3D(256, (5, 5, 5), padding='same', data_format="channels_last")(activ_44)
activ_5 = PReLU()(conv5)
conv52 = Conv3D(256, (5, 5, 5), padding='same', data_format='channels_last')(activ_5)
activ_52 = PReLU()(conv52)
conv53 = Conv3D(256, (5, 5, 5), padding='same', data_format='channels_last')(activ_52)
activ_53  = PReLU()(conv53)

summed_5 = add([activ_53, activ_44])

stride_5 = Conv3DTranspose(256, (2, 2, 2), strides=(2, 2, 2))(summed_5)
activ_54 = PReLU()(stride_5)


#decoding part
#right_layer-4 (6) 256 channels
concat6 = concatenate([summed_4, activ_54])
conv6 = Conv3D(256, (5, 5, 5), padding='same', data_format='channels_last')(concat6)
activ_6 = PReLU()(conv6)
conv62 = Conv3D(256, (5, 5, 5), padding='same', data_format='channels_last')(activ_6)
activ_62 = PReLU()(conv62)
conv63 = Conv3D(256, (5, 5, 5), padding='same', data_format='channels_last')(activ_62)
activ_63 = PReLU()(conv63)

summed_6 = add([activ_63, activ_54]) 

stride_6 = Conv3DTranspose(128, (2, 2, 2), strides=(2, 2, 2))(summed_6)
activ_64 = PReLU()(stride_6)

#right_layer-3 (7) 128 channels
concat7 = concatenate([summed_3, activ_64])
conv7 = Conv3D(128, (5, 5, 5), padding='same', data_format='channels_last')(concat7)
activ_7 = PReLU()(conv7)
conv72 = Conv3D(128, (5, 5, 5), padding='same', data_format='channels_last')(activ_7)
activ_72 = PReLU()(conv72)
conv73 = Conv3D(128, (5, 5, 5), padding='same', data_format='channels_last')(activ_72)
activ_73 = PReLU()(conv73)

summed_7 = add([activ_73, activ_64]) 

stride_7 = Conv3DTranspose(64, (2, 2, 2), strides=(2, 2, 2))(summed_7)
activ_74 = PReLU()(stride_7)

#right_layer-2 (8) 64 channels
concat8 = concatenate([summed_2, activ_74])
conv8 = Conv3D(64, (5, 5, 5), padding='same', data_format='channels_last')(concat8)
activ_8 = PReLU()(conv8)
conv82 = Conv3D(64, (5, 5, 5), padding='same', data_format='channels_last')(activ_8)
activ_82 = PReLU()(conv82)

summed_8 = add([activ_82, activ_74]) 

stride_8 = Conv3DTranspose(32, (2, 2, 2), strides=(2, 2, 2))(summed_8)
activ_83 = PReLU()(stride_8)

#right_layer-1 (9) 32 channels
concat9 = concatenate([summed_1, activ_83])
conv9 = Conv3D(32, (5, 5, 5), padding='same', data_format='channels_last')(concat9)
activ_9 = PReLU()(conv9)

summed_9 = add([activ_9, activ_83]) 

stride_9 = Conv3D(2, (1, 1, 1), padding='same', data_format='channels_last')(summed_9)
activ_92 = PReLU()(stride_9)


#output_layer (is a softmax)
output = Softmax()(activ_92)

model = Model(input_layer, output)


model.summary(line_length=113)



'''def dice_coef(y_true, y_pred):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.reshape(y_pred, (-1, 2))
    intersection = K.mean(y_true_f * y_pred_f[:,0]) + K.mean((1.0 - y_true_f) * y_pred_f[:,1])
    
    return 2. * intersection;

def dice_coef_loss(y_true, y_pred):
    return -dice_coef(y_true, y_pred)
  
from keras import backend as K
'''
'''
def dice_coef(y_true, y_pred, smooth=1):
    """
    Dice = (2*|X & Y|)/ (|X|+ |Y|)
         =  2*sum(|A*B|)/(sum(A^2)+sum(B^2))
    ref: https://arxiv.org/pdf/1606.04797v1.pdf
    """
    intersection = K.sum(K.abs(y_true * y_pred), axis=-1)
    return (2. * intersection + smooth) / (K.sum(K.square(y_true), axis=-1) + K.sum(K.square(y_pred), axis=-1) + smooth)
'''
def dice_coef(y_true, y_pred, smooth=1):
    """
    Dice = (2*|X & Y|)/ (|X|+ |Y|)
         =  2*sum(|A*B|)/(sum(A^2)+sum(B^2))
    ref: https://arxiv.org/pdf/1606.04797v1.pdf
    """
    y_truef = K.flatten(y_true)
    y_predf = K.flatten(y_pred)
    intersection = K.sum((y_true * y_pred), axis=[1, 2, 3])
    return K.mean((2. * intersection + smooth) / (K.sum(K.square(y_true), axis=[1, 2, 3]) + K.sum(K.square(y_pred), axis=[1,2, 3]) + smooth), axis = 0)

def dice_coef_loss(y_true, y_pred):
    return 1-dice_coef(y_true, y_pred)


# Loading Data


path1 = "train_data_p1/"
path2 = "train_data_p2/"
path3 = "train_data_p3/"

def load_images(path, start = 0):
    X = []
    labelmaps = []
    nfiles  = len(os.listdir(path))
    for i in range(start, start+nfiles//4):
        X.append(sitk.ReadImage(path+"Case{:02d}.mhd".format(i)))
        labelmaps.append(sitk.Cast(sitk.ReadImage(path+"Case{:02d}_segmentation.mhd".
                                                   format(i))>0.5, sitk.sitkFloat32))
                                   
    return X, labelmaps


def process_imgs(img_list, dimension, img_size, img_spacing, interp, normdir=False, original = True):
    '''
      Resample all the images so that they have resolution img_size 
      (in v-net = (128, 128, 64)) and spacing is img_spacing (v-net = (1, 1, 1.5)). 
      Interpolation method used is interp (for v-net is linear)
    '''
    resolution = np.zeros(img_size, dtype='float32')
    resam_images = []
    for img in img_list:
        spacing_factor = np.asarray(img.GetSpacing())/img_spacing
        #print("spacing: ", spacing_factor)
        size_factor = np.asarray(img.GetSize()*spacing_factor, dtype='float')
        #print("size_factor: ", size_factor)
        new_size = np.max([size_factor, img_size], axis = 0)
        #needs to be a vector for some reason when using python 3
        #https://github.com/Radiomics/pyradiomics/issues/204
        new_size = new_size.astype(dtype=int).tolist() 
        #print("new_size: ", new_size)
    
        #setting resampler params
        resampler = sitk.ResampleImageFilter()
        resampler.SetReferenceImage(img)
        resampler.SetOutputSpacing(img_spacing)
        resampler.SetSize(new_size)
        resampler.SetInterpolator(interp)
    
        #setting a 3D affine transform
        transform = sitk.AffineTransform(3)
        transform.SetMatrix(img.GetDirection())
        #print("Transform: ", transform)
    
        if normdir:
            resampler.SetTransform(transform.GetInverse())
    
        new_img = resampler.Execute(img)
    
        #getting the region of interest of size exactly equal to img_size desired
   
        centroid = np.asarray(new_size, dtype='float')/2
        #print("centroid", centroid)
        initial_pixel = (centroid - img_size/2).astype('int').tolist()
        #print("initial_pixel: ", initial_pixel)
        roi = sitk.RegionOfInterestImageFilter()
        roi.SetSize(img_size.astype('int').tolist())
        roi.SetIndex(initial_pixel)
    
        new_img_roi = roi.Execute(new_img)
    
        #if the labelmap image, returns equal to 1, where it is bigger than 0.5
        if original:
            resam_images += [np.transpose(sitk.GetArrayFromImage(new_img_roi).astype(dtype=float), [2, 1, 0])]
        else:
            resam_images += [np.transpose(sitk.GetArrayFromImage(new_img_roi).astype(dtype=float), [2, 1, 0]) > 0.5]
    return np.array(resam_images)



filename_imgs = "images.pkl"
filename_labels = "labels.pkl"
if os.path.isfile(filename_imgs) and os.path.isfile(filename_labels):
    with open(filename_imgs, "rb") as img_file:
        new_imags = pickle.load(img_file)
    with open(filename_labels, "rb") as label_file:
        new_labels = pickle.load(label_file)

else:
    imags1, labels1 = load_images(path1)
    imags2, labels2  = load_images(path2, start=26)
    imags3, labels3 = load_images(path3, start=38)
    imags = imags1 + imags2 + imags3
    labels = labels1 + labels2 + labels3
    new_imags = process_imgs(imags, 3, np.array([128, 128, 64]), np.array([1, 1, 1.5]), sitk.sitkLinear)
    new_labels = process_imgs(labels, 3, np.array([128, 128, 64]), np.array([1, 1, 1.5]), sitk.sitkLinear, original=False).astype('int')
    '''new_imags = new_imags.reshape(new_imags.shape + (1,)).astype(np.float32)
    new_labels = new_labels.reshape(new_labels.shape + (1,))
    new_labels = np.concatenate([new_labels, ~new_labels], axis=4)
    new_labels = new_labels.astype(np.float32)'''
    Path(filename_imgs).touch()
    Path(filename_labels).touch()
    with open(filename_imgs, "wb") as img_file:
        pickle.dump(new_imags, img_file)
    with open(filename_labels, "wb") as label_file:
        pickle.dump(new_labels, label_file)


# In[10]:

'''
Test for first time only
with open(filename_imgs, "rb") as img_file:
    test = pickle.load(img_file)
with open(filename_labels, "rb") as label_file:
    testl = pickle.load(label_file)
assert (test == new_imags).all()
assert (testl == new_labels).all()
'''


# Data augmentation extracted from the code in: https://github.com/faustomilletari/VNet/blob/master/utilities.py

def hist_match(source, template):
    """
    Adjust the pixel values of a grayscale image such that its histogram
    matches that of a target image

    Arguments:
    -----------
        source: np.ndarray
            Image to transform; the histogram is computed over the flattened
            array
        template: np.ndarray
            Template image; can have different dimensions to source
    Returns:
    -----------
        matched: np.ndarray
            The transformed output image
    """

    oldshape = source.shape
    source = source.ravel()
    template = template.ravel()

    # get the set of unique pixel values and their corresponding indices and
    # counts
    s_values, bin_idx, s_counts = np.unique(source, return_inverse=True,
                                            return_counts=True)
    t_values, t_counts = np.unique(template, return_counts=True)

    # take the cumsum of the counts and normalize by the number of pixels to
    # get the empirical cumulative distribution functions for the source and
    # template images (maps pixel value --> quantile)
    s_quantiles = np.cumsum(s_counts).astype(np.float64)
    s_quantiles /= s_quantiles[-1]
    t_quantiles = np.cumsum(t_counts).astype(np.float64)
    t_quantiles /= t_quantiles[-1]

    # interpolate linearly to find the pixel values in the template image
    # that correspond most closely to the quantiles in the source image
    #interp_t_values = np.zeros_like(source,dtype=float)
    interp_t_values = np.interp(s_quantiles, t_quantiles, t_values)

    return interp_t_values[bin_idx].reshape(oldshape)

def sitk_show(nda, title=None, margin=0.0, dpi=40):
    figsize = (1 + margin) * nda.shape[0] / dpi, (1 + margin) * nda.shape[1] / dpi

    extent = (0, nda.shape[1], nda.shape[0], 0)
    fig = plt.figure(figsize=figsize, dpi=dpi)
    ax = fig.add_axes([margin, margin, 1 - 2*margin, 1 - 2*margin])

    plt.set_cmap("gray")
    for k in range(0,nda.shape[2]):
        print("printing slice "+str(k))
        ax.imshow(np.squeeze(nda[:,:,k]),extent=extent,interpolation=None)
        plt.draw()
        plt.pause(0.1)
        #plt.waitforbuttonpress()

def computeQualityMeasures(lP,lT):
    quality=dict()
    labelPred=sitk.GetImageFromArray(lP, isVector=False)
    labelTrue=sitk.GetImageFromArray(lT, isVector=False)
    hausdorffcomputer=sitk.HausdorffDistanceImageFilter()
    hausdorffcomputer.Execute(labelTrue>0.5,labelPred>0.5)
    quality["avgHausdorff"]=hausdorffcomputer.GetAverageHausdorffDistance()
    quality["Hausdorff"]=hausdorffcomputer.GetHausdorffDistance()

    dicecomputer=sitk.LabelOverlapMeasuresImageFilter()
    dicecomputer.Execute(labelTrue>0.5,labelPred>0.5)
    quality["dice"]=dicecomputer.GetDiceCoefficient()

    return quality


class DataAug(Sequence):

    def __init__(self, x, y, batch_size):
        self.x = x
        self.y = y
        self.xmod = x
        self.ymod = y
        self.batch_size = batch_size
        self.epoch = 0

    def __len__(self):
        return int(np.ceil(len(self.x) / float(self.batch_size)))

    def __getitem__(self, idx):
        batch_x = np.array(self.xmod[idx * self.batch_size:(idx + 1) * self.batch_size])
        batch_y = np.array(self.ymod[idx * self.batch_size:(idx + 1) * self.batch_size]).astype("int")
        batch_x = batch_x.reshape(batch_x.shape + (1,)).astype(np.float32)
        batch_y = batch_y.reshape(batch_y.shape + (1,))
        batch_y = np.concatenate([batch_y, ~batch_y], axis=4)
        batch_y = batch_y.astype(np.float32)

        return batch_x, batch_y

    def produceRandomlyDeformedImage(self, image, label, numcontrolpoints, stdDef):
        sitkImage=sitk.GetImageFromArray(image, isVector=False)
        sitklabel=sitk.GetImageFromArray(label, isVector=False)

        transfromDomainMeshSize=[numcontrolpoints]*sitkImage.GetDimension()

        tx = sitk.BSplineTransformInitializer(sitkImage,transfromDomainMeshSize)


        params = tx.GetParameters()

        paramsNp=np.asarray(params,dtype=float)
        paramsNp = paramsNp + np.random.randn(paramsNp.shape[0])*stdDef

        paramsNp[0:int(len(params)/3)]=0 #remove z deformations! The resolution in z is too bad

        params=tuple(paramsNp)
        tx.SetParameters(params)

        resampler = sitk.ResampleImageFilter()
        resampler.SetReferenceImage(sitkImage)
        resampler.SetInterpolator(sitk.sitkLinear)
        resampler.SetDefaultPixelValue(0)
        resampler.SetTransform(tx)

        resampler.SetDefaultPixelValue(0)
        outimgsitk = resampler.Execute(sitkImage)
        outlabsitk = resampler.Execute(sitklabel)

        outimg = sitk.GetArrayFromImage(outimgsitk)
        outimg = outimg.astype(dtype=np.float32)

        outlbl = sitk.GetArrayFromImage(outlabsitk)
        outlbl = (outlbl>0.5).astype(dtype=np.float32)

        return outimg,outlbl

    #####FIX
    def produceRandomlyTranslatedImage(self, image, label):
        sitkImage = sitk.GetImageFromArray(image, isVector=False)
        sitklabel = sitk.GetImageFromArray(label, isVector=False)

        itemindex = np.where(label > 0)
        randTrans = (0,np.random.randint(-np.min(itemindex[1])/2,(image.shape[1]-np.max(itemindex[1]))/2),np.random.randint(-np.min(itemindex[0])/2,(image.shape[0]-np.max(itemindex[0]))/2))
        translation = sitk.TranslationTransform(3, randTrans)

        resampler = sitk.ResampleImageFilter()
        resampler.SetReferenceImage(sitkImage)
        resampler.SetInterpolator(sitk.sitkLinear)
        resampler.SetDefaultPixelValue(0)
        resampler.SetTransform(translation)

        outimgsitk = resampler.Execute(sitkImage)
        outlabsitk = resampler.Execute(sitklabel)

        outimg = sitk.GetArrayFromImage(outimgsitk)
        outimg = outimg.astype(dtype=float)

        outlbl = sitk.GetArrayFromImage(outlabsitk) > 0
        outlbl = outlbl.astype(dtype=float)

        return outimg, outlbl

    def on_epoch_end(self):
        #modify train_data here
        self.xmod = []
        self.ymod = []
        print("HERE1")
        for i in range(len(self.x)):
            if np.random.uniform() > 0.5:
                img, label = self.produceRandomlyTranslatedImage(self.x[i], self.y[i])
            elif np.random.uniform() > 0.5:
                img, label = self.produceRandomlyDeformedImage(self.x[i], self.y[i], 2, 0.1)
            else: 
                img = self.x[i]
                label = self.y[i]   
            self.xmod += [img]
            self.ymod += [label]


train_gen = DataAug(new_imags, new_labels,batch_size=2) # you can choose your batch size.


# Compiling model and fitting

# Learning rate update. every 25k iteration decrease by a factor of 10. initial: 10e-4

def step_decay(epoch):
    initial_lrate = 10e-4
    drop = 0.1
    epochs_drop = 25000
    rate = initial_lrate
    if epoch == 25000:
        lrate = initial_lrate * math.pow(drop,  
        np.floor((1+epoch)/epochs_drop))
        rate = lrate
    else: lrate = rate
    return lrate
lrate = LearningRateScheduler(step_decay)

callback_list = [lrate]

model.compile(optimizer=SGD(lr=1e-4, momentum=0.99), loss=dice_coef_loss, metrics=[dice_coef])
model.fit_generator(generator=train_gen, epochs=30000, verbose=1, callbacks=callback_list)


# # Testing 
