# from skimage.io import imread
import numpy as np
import cv2
import math

# import pandas as pd

def get_input(path):

    img = cv2.imread(path)

    return(img)

# def get_output(path,label_file=None):
#
#     img_id = path.split('/')[-1].split('.')[0]
#     img_id = np.int64(img_id)
#     labels = label_file.loc[img_id].values
#
#     return(labels)
#
def preprocess_input(image):

    # >>> Normalize
    image = image / 255.
    image = image.astype('float32')
    # <<< Normalize

    # --- Rescale Image
    # --- Rotate Image
    # --- Resize Image
    # --- Flip Image
    # --- PCA etc.

    return(image)

def get_patches(input_image, patch_size, stride):

    patches = []

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

            patches += [patch]
            patch_array = np.array( patches )
            col_idx_start = col_idx_start + stride
            col_idx_end = col_idx_end + stride

        row_idx_start = row_idx_start + stride
        row_idx_end = row_idx_end + stride
    return patch_array

def image_generator(rgb_files, invert_files, batch_size, patch_size, stride):

    while True:
          # Select files (paths/indices) for the batch
          idx = np.random.choice(np.arange(len(rgb_files)), batch_size, replace=False)

          rgb_batch_paths = np.array(rgb_files)[idx.astype(int)]
          invert_batch_paths = np.array(invert_files)[idx.astype(int)]

          rgb_batch_input = []
          invert_batch_input = []

          # Read in each input, perform preprocessing and get labels
          for rgb_path, invert_path in zip(rgb_batch_paths, invert_batch_paths):

              rgb_input = get_input(rgb_path)
              rgb_input = preprocess_input(rgb_input)
              rgb_patches = get_patches(rgb_input, patch_size, stride)

              invert_input = get_input(invert_path)
              invert_input = preprocess_input(invert_input)
              invert_patches = get_patches(invert_input, patch_size, stride)

              if rgb_batch_input == []:
                  rgb_batch_input = rgb_patches
                  invert_batch_input = invert_patches
              else:
                  rgb_batch_input = np.concatenate((rgb_batch_input, rgb_patches), axis = 0)
                  invert_batch_input = np.concatenate((invert_batch_input, invert_patches), axis = 0)

          yield(rgb_batch_input, invert_batch_input)
