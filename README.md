### Preliminary work

Create following folders:

```
./Hyper
./Hyper/Train_hyper
./Hyper/Train_RGB
./Hyper/Valid_hyper
./Hyper/Valid_RGB
```

Fill the folders with corresponding hyper-spectral (.mat) and RGB images (.bmp).

Currently the hyper-spectral images are obtained from [Foster Dataset](https://personalpages.manchester.ac.uk/staff/d.h.foster/Local_Illumination_HSIs/Local_Illumination_HSIs_2015.html?fbclid=IwAR117m2-52UWDulH1HERqGf6IwP7PH3rPWrMyi7jnGkxVN_JGUOxcITPb1w)

Each hyper-spectral has 33 channels.

You may have to change channel numbers in the neural network input and output shapes to match your images.

`main_hyper.py` is the program you need to run for training.

Build your neural network in function `build_autoencoder`.

Following variables can be changed (and should be fine-tuned) according to your needs

```
patch_size = 100
stride = 100
batch_size_train = 2
batch_size_valid = 1
nb_epoch = 10
```
