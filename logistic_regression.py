# Logistic Regression
# Determine if a student passes (1) or fails (0) based on study hours + tuition

# Import packages
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Preprocess the data

# Read the data file
data = pd.read_csv("logistic_regression_data.csv", header=None)

# Define variables from column vectors
x1 = data.iloc[:,0] # First column training data
x2 = data.iloc[:,1] # Second column training data
x_short = data.iloc[:,0:2] # Training data set (combined)
y = data.iloc[:,2] # Pass/Fail binary output

# Append column of one's to the original data vector to form a n+1 dimensional vector
x0 = np.ones((len(x_short),1)) # Define column vector of 1's
x = np.append(x0,x_short,axis=1) # Add to the n dimensional vector
y = np.array(y).reshape(len(y),1) # reshape y to (len(y)x1) array

# Define hypothesis function
def hypothesis(x, theta):
    z = np.dot(x, theta)
    return 1/(1+np.exp(-z)) - 0.000001 # shift to avoid infinity

# Define cost function
def cost(x, y, h):
    m = len(y)
    firstTerm = -1*(np.dot(np.transpose(y), np.log(hypothesis(x, theta))))
    secondTerm = -1*(np.dot(np.transpose(1-y), np.log(1-hypothesis(x, theta))))
    return((1/m)*(firstTerm+secondTerm))

theta = [[0],[0],[0]]
alpha = 0.0075
m = len(y)
num = 10000000

for i in range(num):
    theta = theta - (alpha / m) * np.dot(np.transpose(x), hypothesis(x, theta) - y)
    if (i%(0.05*num) == 0):
        percentComplete = (i/num)*100 # Print out progress in 5% increments
        print(str(percentComplete)+'%')
#        plt.scatter(i, cost(x, y, hypothesis(x, theta))) # plot cost function

print(theta) # print out parameters
#plt.show() # display cost function plot

# Plot the dataset

# Blue is 0, Green is 1
plt.scatter(x1,x2,c=y,cmap='winter')

# Plot decision line
s1 = np.linspace(4, max(x_short[0]), 100)
s2 = np.linspace(min(x2), max(x2), 100)

eq = (-1*theta[0]-theta[1]*s1)/(theta[2])

plt.plot(s1, eq, color='r')
plt.show()