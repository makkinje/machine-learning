# Logic gates with a simple neural network

# Import packages
import numpy as np
import pandas as pd
import matplotlib as plt


# Define functions for operations

def sigmoid(feature, theta):  # Sigmoid function for logistic
    z = np.dot(feature, theta)
    return 1 / (1 + np.exp(-z))


def orfunction():
    theta = np.array([-10, 20, 20])
    return sigmoid(x, theta)


def andfunction():
    theta = np.array([-30, 20, 20])
    return sigmoid(x, theta)


def norfunction():
    theta = np.array([10, -20, -20])
    return sigmoid(x, theta)


# Loop through binary operations and print

for i in (0, 1):
    for j in (0, 1):
        x = np.array([1, i, j])  # First layer of neural net
        x = np.array([1, andfunction(), norfunction()])  # Second layer of neural net applying logic gate
        # Apply OR operation to compute x1 XNOR x2
        print('x1: ' + str(i) + ' ' + 'x2: ' + str(j) + '     '
              + str(orfunction()))
