# MNIST_Deep-Convolutional_GAN
Neural network generating hand written numbers <br/>

Custom model trained with custom data loader <br/>
# Model
   -  Fully convolutional, no Linear (Fully contected) layers
   -  Generator with 6x TransposeConv, batchnorm, RELu activation
   -  Discriminator with 6x Conv2d, batchnorm, Leaky RElu activation with 0.2 negative_slope
   -  Xavier weight init and zero bias init


# Loss function
Instead of the usual BCE function, I used BCEWithLogitsLoss for better numerical stability. <br/>
Loss formula -<br/>
![Loss](https://github.com/Samuel-Bachorik/MNIST_Deep-Convolutional_GAN/blob/main/Imgs/BCEWithLogits.PNG)

### The course of the loss function
From this figure we can see that 50 epoch is enough for these models. Generator can not get any better. <br/>
For better results you should try wasserstein loss for example. But i think for this aplication this is enough<br/>
<br/>
![LossGraph](https://github.com/Samuel-Bachorik/MNIST_Deep-Convolutional_GAN/blob/main/Imgs/Course%20of%20loss.PNG)

# How to use 
Download dataset here<br/>
[Dataset](https://drive.google.com/file/d/1SfBOq8swmSZf2C1X3HV0cDd08TxkqjNq/view?usp=sharing)<br/>
In `run_training.py` change path to dataset. <br/>
