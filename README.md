# CS-439Project
In this project, we mainly focus on the comparison of scattering model and convolutional neural network. We experiment on different hyperparameters and analyze the best performance of these two models. 
## Model structure
### Scattering model 
Our scattering model consists of one scattering layer and two linear layer. The scattering transformation is a hierarchical signal processing method that captures invariant and stable representations of data. It involves a cascade of Morlet wavelet transforms followed by modulus and averaging operations, resulting in a multi-scale, multi-orientation representation.  Conceptually, a scattering transformation is a series of localized waveforms followed by a non-linear operation, which can be interpreted as a non-trainable CNN.
### Convolutional neural network 
Our convolutional neural network consists of 2 convolution layers and 2 linear layers. The model size is 454922 while the scattering model is 251402. 
## Experiment setup
We vary the optimizer from {adam, rmsprop, sgd}, learning rate from {1e-1, 1e-2, 1e-3}, regularization strength from {0, 5e-4, 5e-3, 5e-2, 5e-1}. We tested each hyperparameter combination under different dataset size : {60, 300, 1200, 6000, 60000}. 
## Results
The final results show that the best convolutional neural network achieves accuracy of 0.9933 in the test set while the best scattering model achieves accuracy of 0.9925. The difference is rather negligible while the scattering model use significantly less parameters. 
## Alternative script
The alt_optml.py is an alternative script which contains a faster version of the scattering transform classifier. It was not used for the report, but is there to illustrate the potential of reducing the training time for the classifiers relying on the scattering transform.
