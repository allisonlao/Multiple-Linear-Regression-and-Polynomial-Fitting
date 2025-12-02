import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path


# Loading the data from the .csv file into a spreadsheet of values
def load():
    data = pd.read_csv("climate_change_dataset.csv")
    return data


# Training/test set split
def train_test_split(x, y, split_ratio=0.8):
    # number of total data points
    n = len(x)

    # shuffle the indices so the split is random
    indices = np.arange(n)
    np.random.shuffle(indices)

    # calculate size of the training set
    training_size = int(split_ratio * n)

    # split the indices---first 80% goes into the training set, the rest goes to the test set
    training_idx = indices[:training_size]
    test_idx = indices[training_size:]

    # return the split datasets
    return x[training_idx], y[training_idx], x[test_idx], y[test_idx]



# Constructing the Design matrix given predictor variables
def construct_design_matrix(x_1, x_2, x_3, x_4, x_5):
    # precip, avg_temp, humidity, solar_irradiance, sea_surface_temp are the column vectors/predictor variables for our data set
    
    # left-most column of ones
    ones = np.ones((len(x_1), 1))

    # stack predictors horizontally, then reshape them to be vertical columns
    X = np.hstack([
        ones,
        x_1.reshape(-1, 1),
        x_2.reshape(-1, 1),
        x_3.reshape(-1, 1),
        x_4.reshape(-1, 1),
        x_5.reshape(-1, 1)
    ])
    return X



# Solving normal equations
def normal_equations_solver(X, y, cond_thresh=1e12):
    # Solve (X^T X) beta = X^T y

    XtX = X.T @ X
    Xty = X.T @ y
    
    beta = np.linalg.solve(XtX, Xty)
    return beta



 # Do least-squares regression via QR factoriaztion
def qr_solver(X, y):
    # Do least-squares regression via QR factoriaztion: X = Q R, R beta = Q^T y
    Q, R = np.linalg.qr(X, mode='reduced')
    beta = np.linalg.solve(R, Q.T @ y)
    return beta



# Compute predicted y hat values from the least-squares estimate
def predict(X, beta):
    return X @ beta


# Compute the squared error
def squared_error(y_actual, y_hat):
    return np.sum((y_actual - y_hat) ** 2)



# Finding singular values (to be used for calculuating condition number)
def singular_values_XtX(X):
    XtX = X.T @ X
    # returns array of singular values in increasing order
    s = np.linalg.svd(XtX, compute_uv=False)
    return s


# Calculating condition number
def condition_number(singular_values):
    sigma_max = singular_values[0]
    sigma_min = singular_values[-1]
    return float(sigma_max / sigma_min)



# Plotting the regression model over the actual points
def plotting(y_actual, y_hat, title='Predicted vs Actual', out_path='pred_vs_actual.png'):
    plt.figure(figsize=(6,6))
    plt.scatter(y_actual, y_hat)
    mn = min(min(y_actual), min(y_hat))
    mx = max(max(y_actual), max(y_hat))
    plt.plot([mn, mx], [mn, mx])
    plt.xlabel('Actual')
    plt.ylabel('Predicted')
    plt.title('CO2 vs Precip, Avg temp, Humidity, Solar Irradiance, Sea Surface Temp')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()
    return out_path
