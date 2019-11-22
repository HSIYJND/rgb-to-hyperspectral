import numpy as np
import matplotlib.pyplot as plt
import cv2
import os
import glob

from custom_load import *

from keras.models import Model, load_model

rgb_train_filelist = glob.glob('./lfw_images/orig_images_test/*.jpg')
inv_train_filelist = glob.glob('./lfw_images/invert_images_test/*.jpg')

model_location = './Trained_Models/lfw_reconstruct-20191121223432/e-029-vl-0.000.h5'
my_model = load_model(model_location)
# opt = Adam(lr=0.0001, beta_1=0.9, beta_2=0.999, decay=0.0001)
# my_model.compile(loss='msle', optimizer=opt, metrics=['accuracy'])

# cv2.namedWindow('Output', cv2.WINDOW_NORMAL)

for rgb_batch, invert_batch in image_generator(sorted(rgb_train_filelist), sorted(inv_train_filelist), batch_size = 1, patch_size = 12, stride = 5):
    num_of_patches = np.shape(rgb_batch)[0]
    for i in range(0, num_of_patches):
        y_output = my_model.predict(rgb_batch)[i]

        plt.figure(1)
        plt.subplot(211)
        plt.imshow(rgb_batch[i,:,:,::-1])

        plt.subplot(212)
        # plt.imshow(invert_batch[12,:,:,::-1])
        plt.imshow(y_output[:,:,::-1])
        plt.show()
