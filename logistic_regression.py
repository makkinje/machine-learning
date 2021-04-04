# Logistic Regression
# Determine if a student passes (1) or fails (0) based on study hours + tuition

# Import packages
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Preprocess the data

# Read the data file

data = pd.read_csv("logistic_regression_data.csv",skiprows=0)

# Define variables from column vectors
x1 = data.iloc[:,0] # Study hour data
x2 = data.iloc[:,1] # Tuition data
y = data.iloc[:,2] # Pass/Fail binary output

# Plot the dataset

# Blue is 0, Green is 1
plt.scatter(x1,x2,c=y,s=500,cmap='winter')
plt.colorbar()
plt.show()

# Now lets build the model, applying gradient descent

theta0 = theta1 = theta2 = 0 # Define parameters
alpha = 0.0002 # Define learning rate
num = 1000000 # Define number of iterations of gradient descent

n = float(len(x1)) # Size of training data

# Hypothesis function

#for i in range(num):
#    inner = np.dot(np.transpose(theta),x)


