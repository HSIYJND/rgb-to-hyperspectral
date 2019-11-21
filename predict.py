import numpy as np
import matplotlib.pyplot as plt
import cv2
import os
import glob

from custom_load import *

from keras.models import Model, load_model

rgb_train_filelist = glob.glob('./lfw_images/orig_images_train/*.jpg')
inv_train_filelist = glob.glob('./lfw_images/invert_images_train/*.jpg')

# rgb_valid_filelist = glob.glob('./all_images/orig_images_valid/*.jpg')
# inv_valid_filelist = glob.glob('./all_images/invert_images_valid/*.jpg')

# model_location = './Trained_Models/MNIST_reconstruct-20191120200854/e-082-vl-0.017.h5'
# model_location = './Trained_Models/lfw_reconstruct-20191121155648_invert/e-099-vl-0.004.h5'
# model_location = './Trained_Models/lfw_reconstruct-20191121160505_orig/e-076-vl-0.004.h5'
model_location = './Trained_Models/lfw_reconstruct-20191121161233_orig_1024/e-093-vl-0.000.h5'
my_model = load_model(model_location)
# opt = Adam(lr=0.0001, beta_1=0.9, beta_2=0.999, decay=0.0001)
# my_model.compile(loss='msle', optimizer=opt, metrics=['accuracy'])

# cv2.namedWindow('Output', cv2.WINDOW_NORMAL)

for rgb_batch, invert_batch in image_generator(sorted(rgb_train_filelist), sorted(inv_train_filelist), batch_size = 1):
    y_output = my_model.predict(rgb_batch)[0]

    plt.figure(1)
    plt.subplot(211)
    plt.imshow(rgb_batch[0,:,:,::-1])

    plt.subplot(212)
    plt.imshow(y_output[:,:,::-1])
    plt.show()
