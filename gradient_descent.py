# Linear Regression with Gradient Descent

# Import packages
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Set parameters for figure
plt.rcParams['figure.figsize'] = (12.0, 9.0)

# Preprocessing input data

# Read the file with pandas
data = pd.read_csv('linear_regression_data.csv')

# Define variables from columns in the dataset. iloc integer based position.
x = data.iloc[:,0] # 0 column is x
y = data.iloc[:,1] # 1 column is y

# Plot the dataset with matplotlib
plt.scatter(x,y)
plt.show()

# Now lets build the model

theta0 = theta1 = 0 # define linear equation parameters, set to zero

alpha = 0.0002 # learning rate
num = 1000000 # number of iterations to perform gradient descent

n = float(len(x)) # number of elements in x

# Perform gradient descent

for i in range(num):
    htheta = theta0 + theta1*x # hypothesis function

    d0 = 1/n*(sum(htheta-y)) # derivative with respect to j=0
    d1 = 1/n*(sum(x*(htheta-y))) # derivative with respect to j=1

    temp0 = theta0
    temp1 = theta1

    theta0 = theta0 - alpha*d0
    theta1 = theta1 - alpha*d1

print(theta0, theta1)

plt.scatter(x,y)
plt.plot([min(x), max(x)], [min(htheta), max(htheta)], color='red')  # regression line
plt.show()
