import numpy as np
import cv2
import os
import glob

from custom_load import *

def show_images(images, cols = 1, titles = None):
    """Display a list of images in a single figure with matplotlib.

    Parameters
    ---------
    images: List of np.arrays compatible with plt.imshow.

    cols (Default = 1): Number of columns in figure (number of rows is
                        set to np.ceil(n_images/float(cols))).

    titles: List of titles corresponding to each image. Must have
            the same length as titles.
    """
    assert((titles is None)or (len(images) == len(titles)))
    n_images = len(images)
    if titles is None: titles = ['Image (%d)' % i for i in range(1,n_images + 1)]
    fig = plt.figure()
    for n, (image, title) in enumerate(zip(images, titles)):
        a = fig.add_subplot(cols, np.ceil(n_images/float(cols)), n + 1)
        if image.ndim == 2:
            plt.gray()
        plt.imshow(image)
        a.set_title(title)
    fig.set_size_inches(np.array(fig.get_size_inches()) * n_images)
    plt.show()

# from keras.models import Model, load_model

filenames_rgb   = glob.glob('./Hyper/Train_RGB/*.bmp')
filenames_hyper = glob.glob('./Hyper/Train_hyper/*.mat')

# model_location = './Trained_Models/lfw_reconstruct-20191121223432/e-029-vl-0.000.h5'
# my_model = load_model(model_location)

# opt = Adam(lr=0.0001, beta_1=0.9, beta_2=0.999, decay=0.0001)
# my_model.compile(loss='msle', optimizer=opt, metrics=['accuracy'])

# cv2.namedWindow('Output', cv2.WINDOW_NORMAL)

# for rgb_batch, invert_batch in image_generator(sorted(rgb_train_filelist), sorted(inv_train_filelist), batch_size = 1, patch_size = 12, stride = 5):
#     num_of_patches = np.shape(rgb_batch)[0]
#     for i in range(0, num_of_patches):
#         y_output = my_model.predict(rgb_batch)[i]
#
#         plt.figure(1)
#         plt.subplot(211)
#         plt.imshow(rgb_batch[i,:,:,::-1])
#
#         plt.subplot(212)
#         # plt.imshow(invert_batch[12,:,:,::-1])
#         plt.imshow(y_output[:,:,::-1])
#         plt.show()

patch_size = 256
stride = 256
patch_batch_size = 10

for rgb_batch, hyper_batch in nested_image_generator(filenames_rgb, filenames_rgb, patch_batch_size, patch_size, stride):
    images = np.concatenate((rgb_batch, hyper_batch), axis=0)
    images_list = [images[i,:,:,:] for i in range(np.shape(images)[0])]
    show_images(images_list, cols = 2, titles = None)


# for rgb_batch, hyper_batch in image_generator_with_hyper(sorted(filenames_rgb), sorted(filenames_hyper), batch_size = 1, patch_size = patch_size, stride = stride):
#     show_hyper_patches(hyper_batch, patch_size)
