# Linear Regression with the Normal Equation

# Import packages
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Set parameters for figure
plt.rcParams['figure.figsize'] = (12.0, 9.0)

# Preprocessing input data

# Read the file with pandas
data = pd.read_csv('linear_regression_data.csv', header=None)

# Define variables from columns in the dataset. iloc integer based position.
# np.array takes in the data as row vectors, transpose them to column vectors.
x = np.transpose(np.array([data.iloc[:,0]])) # 0 column is x
y = np.transpose(np.array([data.iloc[:,1]])) # 1 column is y

# Plot the dataset with matplotlib
plt.scatter(x,y)
plt.show()

# Define x0 vector with one's
x0 = np.ones((len(x),1))

# Append it to the original data vector to form a n+1 dimensional vector
x_long = np.append(x0,x,axis=1)

# Now lets build the model

# Define theta as n dimensional column vector and use normal equation

theta = np.dot(np.dot(np.linalg.inv(np.dot(np.transpose(x_long),x_long)),np.transpose(x_long)),y) # (x^T*x)^-1*x^T*y
print(theta)

# Define linear equation with theta parameters

htheta = theta[0] + theta[1]*x

# Plot regression line
plt.scatter(x,y)
plt.plot([min(x), max(x)], [min(htheta), max(htheta)], color='red')  # regression line
plt.show()
