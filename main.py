import numpy as np
import matplotlib.pyplot as plt
import cv2
import os
import glob

from custom_load import *

from keras.layers import Dense, Flatten, Reshape, Input, InputLayer
from keras.models import Sequential, Model

from datetime import datetime

from keras.callbacks import ModelCheckpoint
from keras.callbacks import CSVLogger

def build_autoencoder(img_shape, code_size):
    # The encoder
    encoder = Sequential()
    encoder.add(InputLayer(img_shape))
    encoder.add(Flatten())
    encoder.add(Dense(code_size))

    # The decoder
    decoder = Sequential()
    decoder.add(InputLayer((code_size,)))
    decoder.add(Dense(np.prod(img_shape))) # np.prod(img_shape) is the same as 32*32*3, it's more generic than saying 3072
    decoder.add(Reshape(img_shape))

    return encoder, decoder

rgb_train_filelist = glob.glob('./lfw_images/orig_images_train/*.jpg')
inv_train_filelist = glob.glob('./lfw_images/invert_images_train/*.jpg')

rgb_valid_filelist = glob.glob('./lfw_images/orig_images_valid/*.jpg')
inv_valid_filelist = glob.glob('./lfw_images/invert_images_valid/*.jpg')

# for rgb_batch, invert_batch in image_generator(sorted(rgb_train_filelist), sorted(inv_train_filelist), batch_size = 5):
#     print(np.shape(rgb_batch))
#     print("done")
    # plt.figure(1)
    # plt.subplot(211)
    # plt.imshow(rgb_batch[0,:,:])
    #
    # plt.subplot(212)
    # plt.imshow(invert_batch[0,:,:])
    # plt.show()

# Same as (32,32,3), we neglect the number of instances from shape
# IMG_SHAPE = X.shape[1:]

# >>> Saves the model weights after each epoch if the validation loss decreased
output_dir = './Trained_Models/'
now = datetime.now()
nowstr = now.strftime('lfw_reconstruct-%Y%m%d%H%M%S')
now = os.path.join(output_dir, nowstr)
# <<< Saves the model weights after each epoch if the validation loss decreased

# >>> Make the directory
os.makedirs( now, exist_ok=True )
# <<< Make the directory

# >>> Create our callbacks
savepath = os.path.join( now, 'e-{epoch:03d}-vl-{val_loss:.3f}.h5' )
savepath_log = os.path.join( now, 'CSV_Log.csv' )

checkpointer = ModelCheckpoint(filepath=savepath, monitor='val_loss', mode='min', verbose=0, save_best_only=True)
# checkpointer = ModelCheckpoint(filepath=savepath, monitor='val_acc', mode='max', verbose=0, save_weights_only=True, save_best_only=True)
csv_logger = CSVLogger(savepath_log, append=True, separator=';')
# tb = TensorBoard(log_dir='./logs', histogram_freq=1, write_graph=True, write_images=True)

callbacks_list = [checkpointer, csv_logger]
# <<< Create our callbacks


patch_size = 12
stride = 5

IMG_SHAPE = [patch_size, patch_size, 3]
# IMG_SHAPE = np.shape(one_image)

encoder, decoder = build_autoencoder(IMG_SHAPE, 1024)

inp = Input(IMG_SHAPE)
code = encoder(inp)
reconstruction = decoder(code)

autoencoder = Model(inp,reconstruction)
autoencoder.compile(optimizer='adamax', loss='mse')

print(autoencoder.summary())

batch_size_train = 50
batch_size_valid = 20
nb_epoch = 100

one_image = cv2.imread(rgb_train_filelist[0])
one_patch = get_patches(one_image, patch_size, stride)
patches_obtained = np.shape(one_patch)[0]

print('Size of patches obtained: ', patches_obtained)
# history = autoencoder.fit_generator(image_generator(sorted(rgb_train_filelist), sorted(inv_train_filelist), batch_size = batch_size_train),
#                 validation_data=image_generator(sorted(rgb_valid_filelist), sorted(inv_valid_filelist), batch_size = batch_size_valid),
#                 validation_steps=len(rgb_valid_filelist) / batch_size_valid,
#                 steps_per_epoch=len(rgb_train_filelist) / batch_size_train,
#                 epochs=nb_epoch, verbose=1, callbacks=callbacks_list)

history = autoencoder.fit_generator(image_generator(sorted(rgb_train_filelist), sorted(rgb_train_filelist), batch_size = batch_size_train, patch_size = patch_size, stride = stride),
                steps_per_epoch=len(rgb_train_filelist) / batch_size_train,
                validation_data=image_generator(sorted(rgb_valid_filelist), sorted(rgb_valid_filelist), batch_size = batch_size_valid, patch_size = patch_size, stride = stride),
                validation_steps=len(rgb_valid_filelist) / batch_size_valid,
                epochs=nb_epoch, verbose=1, callbacks=callbacks_list)

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
