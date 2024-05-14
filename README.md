# KAN-network-from-scratch
Building a customisable, general-purpose, versatile Kolomogorov-Arnold Network from basic, explicit Python commands without external libraries.

To execute: run GUImain.py

I will allow myself numpy if necessary, as well as multiprocessing to speed up training

Currently under construction. 

Eventually this can be deployed onto GPUs via pytorch.cuda - or maybe not, due to the nonlinearity of the parameters...

Neural Network Model Modules:
turboKAN: a simple, working, optimised KAN network built according to the paper's specifications. However, it runs slowly due to the splines...
Catherine: Instead of arbitrary splines and grids, Catherine uses a single cubic polynomial in single grid in each activation function. The coefficients of the polynomial and the Silu(x) are trainable.
Katalina: Implements smaller batch training on top of Catherine - split your large dataset into a few smaller, manageable sub-batches. Problem: very quick descent of errors in early stage but very slow in later stages. 
Lerochka: Implementing RMSprop and Momentum to accelerate training and escape from local minima.  (under construction)

Idea: translate Python into C++ for faster performance?
