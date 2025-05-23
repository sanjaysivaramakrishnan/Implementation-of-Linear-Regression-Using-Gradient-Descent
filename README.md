# Implementation-of-Linear-Regression-Using-Gradient-Descent

## AIM:
To write a program to predict the profit of a city using the linear regression model with gradient descent.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
### Algorithm: Linear Regression with Gradient Descent

1. **Load the dataset** from CSV containing startup data.
2. **Extract features `X` and target `y`**, convert both to float, and reshape `y` to a column vector.
3. **Scale the features `X`** using `StandardScaler` to normalize the data.
4. **Initialize model parameters**:
   - Weights `w` as a zero vector.
   - Bias `b` as zero.
5. **Train the model using gradient descent** for a fixed number of iterations:
   - Compute predictions using `Xw + b`.
   - Calculate errors and update `w` and `b` using gradient descent rules.
6. **Prepare new input data**:
   - Scale the new data using the previously fitted scaler.
   - Predict the output using the trained model.
7. **Output the predicted value**.

## Program:
```
/*
Program to implement the linear regression using gradient descent.
Developed by: Sanjay Sivaramakrishnan M
RegisterNumber:  212223240151

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
def linear_regression(X1, y, learning_rate=0.01, num_iters=1000):
    # Add a column of ones to X for the intercept term
  X = np.c_[np.ones(len(X1)), X1]
  # Initialize theta with zeros
  theta = np.zeros(X.shape[1]).reshape(-1,1)
  # Perform gradient descent
  for _ in range(num_iters):
    # Calculate predictions
    predictions = (X).dot(theta).reshape(-1, 1)
    # Calculate errors
    errors = (predictions - y).reshape(-1,1)
    # Update theta using gradient descent
    theta -= learning_rate * (1 / len(X1)) * X.T.dot(errors)
  return theta
data = pd.read_csv(r'C:\Users\admin\Desktop\Python_jupyter\ML LEARN\intro_machine_learning\data_sets\50_Startups.csv',header=None)
print(data.head())

# Assuming the last column is your target variable 'y' and the preceding columns are your features 'X'
X = (data.iloc[1:, :-2].values)
print(X)
X1=X.astype(float)
scaler = StandardScaler()
y = (data.iloc[1:,-1].values).reshape(-1,1)
print(y)
X1_Scaled = scaler.fit_transform(X1)
Y1_Scaled = scaler.fit_transform(y)
print(X1_Scaled)
print(Y1_Scaled)

# Learn model parameters
theta = linear_regression(X1_Scaled, Y1_Scaled)

# Predict target value for a new data point
new_data = np.array([165349.2,136897.8,471784.1]).reshape(-1,1)
new_Scaled = scaler.fit_transform(new_data)
prediction =np.dot(np.append(1, new_Scaled), theta)
prediction=prediction.reshape(-1,1)
pre=scaler.inverse_transform(prediction)
print(f"Predicted value: {pre}")

*/
```

## Output:
![image](https://github.com/user-attachments/assets/1365d61c-5579-44b9-b3c3-9adf355f5071)
![image](https://github.com/user-attachments/assets/f4e676d1-5c0f-4855-8b3c-b3441669ec67)


## Result:
Thus the program to implement the linear regression using gradient descent is written and verified using python programming.
