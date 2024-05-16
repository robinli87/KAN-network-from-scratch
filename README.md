# KAN-network-from-scratch
Building a customisable, general-purpose, versatile Kolomogorov-Arnold Network from basic, explicit Python commands without external libraries (except the GUI). This is an explicit reconstruction of KAN network modules and allows you to understand the inner mechanisms involved. They can also be deployed for some simple uses such as approximating a function with some given data. 

To execute: python3 main.py

Dependencies:
* pyqt5
* matplotlib
* numpy
If you cannot install QT dependencies, you can use the Tkinter or CLI main routines in the test folder. 

Enter the necessary inputs as specified. The data used to train the network are in model_inputs.csv and model_outputs.csv, where input is the input feature vector flattened into a csv file with each grid representing a component. The output data file for training is formatted in a similar way. test_inputs.csv should contain unseen data to evaluate the model's performance. By default, sample points of the function f(x) = exp( sin(x1^2 + x2^2) + sin(x3^2 + x4^2) ) are being used to train. This function is used in the paper by Ziming Liu et al. 

![training paper data](https://github.com/robinli87/KAN-network-from-scratch/assets/101805462/ff2958b5-6578-40e9-9345-029864b492e2)

Neural Network Model Modules:
* KAN_optimised: a simple, working, optimised KAN network built according to the paper's specifications. Normalises the output at each layer. However, it runs slowly due to the splines...
* TurboKAN: KAN_optimised but without normalisation and some additional optimisations. Trains slightly faster.
* Catherine: Instead of arbitrary splines and grids, Catherine uses a single cubic polynomial in single grid in each activation function. The coefficients of the polynomial and the Silu(x) are trainable. This allows you to train much larger networks at descent speeds.
* Katalina: Implements smaller batch training on top of Catherine - split your large dataset into a few smaller, manageable sub-batches. Problem: very quick descent of errors in early stage but very slow in later stages. 

Lerochka: Implementing RMSprop and Momentum to accelerate training and escape from local minima.  (under construction)

Idea: translate Python into C++ for faster performance?
