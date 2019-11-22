import scipy.io
import cv2
import glob
import os
import matplotlib.pyplot as plt
import numpy as np

filenames_hyper = glob.glob('./Train_hyper/*.mat')
filenames_rgb   = glob.glob('./Train_RGB/*.bmp')

filenames_hyper.sort()
filenames_rgb.sort()

print('Number of bmp images: ', np.size(filenames_rgb))
for i in range(0, len(filenames_hyper)):
    print([filenames_hyper[i], filenames_rgb[i]])
    # load hyperspectral image
    mat =  scipy.io.loadmat(filenames_hyper[i])
    hyper = np.float32(mat['hsi'])

    # load color image
    rgb =  cv2.imread(filenames_rgb[i])
    rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)

    print('Shape of hyper image: ', np.shape(hyper))
    plt.imshow(rgb)
    plt.show()
