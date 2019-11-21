# from skimage.io import imread
import numpy as np
import cv2
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

    image = image / 255.
    image = image.astype('float32')

    # --- Rescale Image
    # --- Rotate Image
    # --- Resize Image
    # --- Flip Image
    # --- PCA etc.

    return(image)

def image_generator(rgb_files, invert_files, batch_size):

    while True:
          # Select files (paths/indices) for the batch
          idx = np.random.choice(np.arange(len(rgb_files)), batch_size, replace=False)

          rgb_batch_paths = np.array(rgb_files)[idx.astype(int)]
          invert_batch_paths = np.array(invert_files)[idx.astype(int)]
          # print(idx[0])
          # print(rgb_files[0])
          # print(invert_files[0])

          rgb_batch_input = []
          invert_batch_input = []
          # batch_output = []

          # Read in each input, perform preprocessing and get labels
          for rgb_path, invert_path in zip(rgb_batch_paths, invert_batch_paths):
              rgb_input = get_input(rgb_path)
              rgb_input = preprocess_input(rgb_input)
              invert_input = get_input(invert_path)
              invert_input = preprocess_input(invert_input)
              # output = get_output(input_path,label_file=label_file )

              # input = preprocess_input(image=input)
              rgb_batch_input += [ rgb_input ]
              invert_batch_input += [ invert_input ]
              # batch_output += [ output ]
          # Return a tuple of (input,output) to feed the network
          rgb_batch = np.array( rgb_batch_input )
          invert_batch = np.array( invert_batch_input )
          # batch_y = np.array( batch_output )

          yield(rgb_batch, invert_batch)
