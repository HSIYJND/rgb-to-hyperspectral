import numpy as np
import pandas as pd
import tarfile
import tqdm
import cv2
import os

# http://www.cs.columbia.edu/CAVE/databases/pubfig/download/lfw_attributes.txt
ATTRS_NAME = "lfw_attributes.txt"

# http://vis-www.cs.umass.edu/lfw/lfw-deepfunneled.tgz
IMAGES_NAME = "lfw-deepfunneled.tgz"

# http://vis-www.cs.umass.edu/lfw/lfw.tgz
RAW_IMAGES_NAME = "lfw.tgz"

def decode_image_from_raw_bytes(raw_bytes):
    img = cv2.imdecode(np.asarray(bytearray(raw_bytes), dtype=np.uint8), 1)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img

def load_lfw_dataset(
        use_raw=False,
        dx=80, dy=80,
        dimx=45, dimy=45):

    # Read attrs
    df_attrs = pd.read_csv(ATTRS_NAME, sep='\t', skiprows=1)
    df_attrs = pd.DataFrame(df_attrs.iloc[:, :-1].values, columns=df_attrs.columns[1:])
    imgs_with_attrs = set(map(tuple, df_attrs[["person", "imagenum"]].values))

    # Read photos
    all_photos = []
    photo_ids = []

    # tqdm in used to show progress bar while reading the data in a notebook here, you can change
    # tqdm_notebook to use it outside a notebook
    with tarfile.open(RAW_IMAGES_NAME if use_raw else IMAGES_NAME) as f:
        for m in tqdm.tqdm_notebook(f.getmembers()):
            # Only process image files from the compressed data
            if m.isfile() and m.name.endswith(".jpg"):
                # Prepare image
                img = decode_image_from_raw_bytes(f.extractfile(m).read())

                # Crop only faces and resize it
                img = img[dy:-dy, dx:-dx]
                img = cv2.resize(img, (dimx, dimy))

                # Parse person and append it to the collected data
                fname = os.path.split(m.name)[-1]
                fname_splitted = fname[:-4].replace('_', ' ').split()
                person_id = ' '.join(fname_splitted[:-1])
                photo_number = int(fname_splitted[-1])
                if (person_id, photo_number) in imgs_with_attrs:
                    all_photos.append(img)
                    photo_ids.append({'person': person_id, 'imagenum': photo_number})

    photo_ids = pd.DataFrame(photo_ids)
    all_photos = np.stack(all_photos).astype('uint8')

    # Preserve photo_ids order!
    all_attrs = photo_ids.merge(df_attrs, on=('person', 'imagenum')).drop(["person", "imagenum"], axis=1)

    return all_photos, all_attrs

def main():
    dataset_images, attr = load_lfw_dataset(use_raw=True, dimx=32, dimy=32)

    image_size = np.shape(dataset_images)[1]

    total_number_of_images = np.shape(dataset_images)[0]
    train_num_images = int((total_number_of_images * 60) / 100)
    valid_test_num_images = int((total_number_of_images - train_num_images) / 2)

    orig_train_folder = './lfw_images/orig_images_train/'
    orig_valid_folder = './lfw_images/orig_images_valid/'
    orig_test_folder  = './lfw_images/orig_images_test/'

    invert_train_folder = './lfw_images/invert_images_train/'
    invert_valid_folder = './lfw_images/invert_images_valid/'
    invert_test_folder  = './lfw_images/invert_images_test/'

    os.makedirs(orig_train_folder, exist_ok=True )
    os.makedirs(orig_valid_folder, exist_ok=True )
    os.makedirs(orig_test_folder,  exist_ok=True )

    os.makedirs(invert_train_folder, exist_ok=True )
    os.makedirs(invert_valid_folder, exist_ok=True )
    os.makedirs(invert_test_folder,  exist_ok=True )

    cv2.namedWindow('Image', cv2.WINDOW_NORMAL)

    for i in range(0, train_num_images):
        # orig_image = np.asarray(data[i]).squeeze()
        orig_image = dataset_images[i,:,:,::-1]
        invert_image = cv2.bitwise_not(orig_image)
        orig_image_filename   = orig_train_folder + 'orig_train_' + str(i) + '.jpg'
        invert_image_filename = invert_train_folder + 'invert_train_' + str(i) + '.jpg'

        cv2.imwrite(orig_image_filename, orig_image)
        cv2.imwrite(invert_image_filename, invert_image)

        cv2.imshow('Image', orig_image)
        cv2.waitKey(1)

    for i in range(0, valid_test_num_images):
        orig_image = dataset_images[i,:,:,::-1]
        invert_image = cv2.bitwise_not(orig_image)
        orig_image_filename   = orig_valid_folder + 'orig_valid_' + str(i) + '.jpg'
        invert_image_filename = invert_valid_folder + 'invert_valid_' + str(i) + '.jpg'

        cv2.imwrite(orig_image_filename, orig_image)
        cv2.imwrite(invert_image_filename, invert_image)

        cv2.imshow('Image', invert_image)
        cv2.waitKey(1)

    for i in range(0, valid_test_num_images):
        orig_image = dataset_images[i,:,:,::-1]
        invert_image = cv2.bitwise_not(orig_image)
        orig_image_filename   = orig_test_folder + 'orig_test_' + str(i) + '.jpg'
        invert_image_filename = invert_test_folder + 'invert_test_' + str(i) + '.jpg'

        cv2.imwrite(orig_image_filename, orig_image)
        cv2.imwrite(invert_image_filename, invert_image)

        cv2.imshow('Image', orig_image)
        cv2.waitKey(1)

if __name__== '__main__':
  main()
