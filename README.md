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
