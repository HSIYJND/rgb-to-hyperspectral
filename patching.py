import numpy as np
import matplotlib.pyplot as plt
import cv2
import os
import glob
import math

input_image = cv2.imread('/home/osama/Programs/one_more/lfw_images/orig_images_train/orig_train_0.jpg')

rgb_patches = []

patch_size = 15
stride = 5

input_height = np.shape(input_image)[0]
input_width = np.shape(input_image)[1]
image_channels = np.shape(input_image)[2]

row_idx_start = 0
row_idx_end = patch_size
padding_row = 0
row_flag = True

while row_flag:
    col_flag = True
    col_idx_start = 0
    col_idx_end = patch_size
    padding_col = 0

    if row_idx_end > input_height:
        padding_row = row_idx_end - input_height
        row_flag = False

    while col_flag:
        if col_idx_end > input_width:
            padding_col = col_idx_end - input_width
            col_flag = False
        patch = input_image[row_idx_start:row_idx_end - padding_row, col_idx_start:col_idx_end - padding_col, :]
        patch = np.pad(patch, [(0, padding_row), (0, padding_col), (0, 0)], mode='constant')

        rgb_patches += [patch]

        col_idx_start = col_idx_start + stride
        col_idx_end = col_idx_end + stride

        cv2.imshow("patch", patch)
        cv2.waitKey(0)

    row_idx_start = row_idx_start + stride
    row_idx_end = row_idx_end + stride

print('number of patches: ', len(rgb_patches))
