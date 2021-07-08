### gan.py

Contains implementations of various GAN models for use in time-series AD

### build_model.py

Contains functions for building various AD scoring methods (including the GANs). Serves as a blueprint
for what parameters the various models requires

### The remainding files

The remainding files are example implementation of various pytorch models one can give the GANs as arguments.
Most are either autoencoders, or encoder-based discriminators. The file tcn.py contains an implementation of 
Temporal Convolutional Networks, and uses them as building blocks of an autoencoder.