# Image_recognisition-ANN

## Description
This directory contains the implementation of machine learning models using Python. Specifically, it contains the implementation of a Multilayer Perceptron (MLP), Convolutional Neural Network (CNN), and Residual Network (ResNet) in the files `mlp.py`, `cnn.py`, and `resnet.py`, respectively. 

The code requires several libraries to be installed. These are listed in the `requirements.txt` file. To install these libraries, simply run `make` in the terminal.

## Usage
To use the code, simply run one of the three scripts:
- `python3 mlp.py [operation: -load/-save]`: Runs the MLP implementation.
- `python3 cnn.py[operation : -load/-save]`: Runs the CNN implementation.
- `python3 resnet.py[operation:-load/-save]`: Runs the ResNet implementation.

-save : trains the data and load 
-load : loads params of trained data , here we only included the best test accuracy
