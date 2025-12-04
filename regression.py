import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# Loading the data from the .csv file into a spreadsheet of values
def load():
    data = pd.read_csv("climate_change_dataset.csv")
    return data


# Train / test split
def train_test_split(x, y, split_ratio=0.8, seed=0):
    # number of total data points
    n = len(x)

    # create rng with a fixed seed, so it generates the same thing each time
    rng = np.random.default_rng(seed)

    # shuffle the indices so the split is random (but repeatable)
    indices = np.arange(n)
    rng.shuffle(indices)

    # calculate size of the training set
    training_size = int(split_ratio * n)

    # split the indices -first 80% goes into the training set, the rest goes to the test set
    training_idx = indices[:training_size]
    test_idx = indices[training_size:]

    # return the split datasets
    return x[training_idx], y[training_idx], x[test_idx], y[test_idx]


# Constructing simple design matrix given 5 predictor variables
def construct_design_matrix(x_1, x_2, x_3, x_4, x_5):
    # precip, avg_temp, humidity, solar_irradiance, sea_surface_temp are the column vectors/predictor variables for our data set
    
    # left-most column of ones
    ones = np.ones((len(x_1), 1))

    # stack predictors and column of ones into the design matrix X
    X = np.hstack([
        ones,
        x_1.reshape(-1, 1),
        x_2.reshape(-1, 1),
        x_3.reshape(-1, 1),
        x_4.reshape(-1, 1),
        x_5.reshape(-1, 1)
    ])
    return X



# Constructing a polynomial design matrix (for one predictor)
def construct_poly_design_matrix(x, degree):
    #builds polynomial design matrix [1, x, x^2, ..., x^degree] for a single predictor x
    
    x_arr = np.asarray(x).reshape(-1)
    n = len(x_arr)
    
    X = np.ones((n, degree + 1))

    
    for d in range(1, degree + 1):
        X[:, d] = x_arr ** d
        
    return X


# Solving normal equations given design matrix X and response vector y
def normal_equations_solver(X, y):
    # solves (X^T X) beta = X^T y

    XtX = X.T @ X
    Xty = X.T @ y
    
    beta = np.linalg.solve(XtX, Xty)

    return beta



# Compute predicted y hat values from the least-squares estimate, beta
def predict(X, beta):
    return X @ beta
    


# Compute the squared error
def squared_error(y_actual, y_hat):
    return np.sum((y_actual - y_hat) ** 2)



# Finding singular values (to be used for calculuating condition number)
def singular_values_XtX(X):
    XtX = X.T @ X
    
    # returns array of singular values in decreasing order
    s = np.linalg.svd(XtX, compute_uv=False)
    return s



# Calculating condition number
def condition_number(singular_values):
    sigma_max = singular_values[0]
    sigma_min = singular_values[-1]
    return float(sigma_max / sigma_min)



# Plotting predicted values from regression model versus actual values
def plotting(y_actual, y_hat, title='Pred vs Actual', returned ='pred_vs_actual.png'):

    
    y_hat = np.asarray(y_hat).reshape(-1)
    y_actual = np.asarray(y_actual).reshape(-1)
    
    plt.figure(figsize=(6,6))
    plt.scatter(y_actual, y_hat, s=20)

    
    mn = min(y_actual.min(), y_hat.min())
    mx = max(y_actual.max(), y_hat.max())

    
    plt.plot([mn, mx], [mn, mx], linestyle='--', color='gray')

    plt.xlabel('Actual')
    plt.ylabel('Predicted')
    plt.title(title)
    plt.grid(True)
    plt.tight_layout()

    #saves the plot as a .png in the folder
    plt.savefig(returned)
    plt.close()
    
    return returned


# Plotting Training vs Test Error
def plot_errors(train_errs, test_errs, title = 'Training vs Test Error' , returned = 'test_vs_training_errors.png'):
    degrees = np.arange(1, len(train_errs) + 1)
    plt.figure(figsize=(7,5))
    plt.plot(degrees, train_errs, marker='o', label='Training Error (SE)')
    plt.plot(degrees, test_errs, marker='o', label='Test Error (SE)')
    plt.xlabel('Polynomial Degree')
    plt.ylabel('Squared Error')
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(returned)
    plt.close()
    return returned


# Plotting different polynomial fits (degrees 1 to 5)
def plot_polynomial_fits(x, y, max_degree=5):
    
    x = np.asarray(x).reshape(-1)
    y = np.asarray(y).reshape(-1)
    plt.figure(figsize=(8,6))
    plt.scatter(x, y, color='gray', s=15, label='Data')
    x_grid = np.linspace(x.min(), x.max(), 500)

    #plotting each polynomial model
    for deg in range(1, max_degree + 1):
        X_train = construct_poly_design_matrix(x, deg)
        beta = normal_equations_solver(X_train, y)
        X_grid = construct_poly_design_matrix(x_grid, deg)
        y_grid = predict(X_grid, beta)
        plt.plot(x_grid, y_grid, label=f"deg {deg}")
        
    plt.xlabel('Predictor (x)')
    plt.ylabel('Response (y)')
    plt.title('Polynomial fits of different degrees')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('polynomial_fits.png')
    plt.close()
    return 'polynomial_fits.png'



# Polynomial regression analysis, combines the previously defined functions and returns polynomial model regression analysis 
def polynomial_regression_analysis(x, y, max_degree=10):
    """
    For degrees 1 to max_degree, this builds polynomial design matrices, does the polynomial regression on the training and test sets, 
    computes training & test squared error, computes singular values & condition number of X^T X on the training set, 
    and returns the lists: train_errors, test_errors, cond_numbers, singular_values_list
    """
    
    # Make the train/test sets
    X_all = np.asarray(x).reshape(-1)
    y_arr = np.asarray(y).reshape(-1)
    X_tr, y_tr, X_te, y_te = train_test_split(X_all, y_arr)
    train_errors = []
    test_errors = []
    cond_numbers = []
    svals_list = []

    # do polynomial regression for degrees 1 through max_degree, computes squared error and condition number
    for deg in range(1, max_degree + 1):
        X_train = construct_poly_design_matrix(X_tr, deg)
        X_test = construct_poly_design_matrix(X_te, deg)
        beta = normal_equations_solver(X_train, y_tr)
        y_tr_hat = predict(X_train, beta)
        y_te_hat = predict(X_test, beta)
        train_errors.append(squared_error(y_tr, y_tr_hat))
        test_errors.append(squared_error(y_te, y_te_hat))
        svals = np.linalg.svd(X_train.T @ X_train, compute_uv=False)
        svals_list.append(svals)
        cond_numbers.append(condition_number(svals))

    # returns the training set errors, test set errors, condition numbers, and singular values
    return {
        'train_errors': train_errors,
        'test_errors': test_errors,
        'cond_numbers': cond_numbers,
        'svals_list': svals_list
    }


def main():

    csv_path = Path("climate_change_dataset.csv")

    #makes sure the file is in the directory
    if not csv_path.exists():
        print("Missing the climate_change_dataset.csv file, please put it in the same folder so this can run")
        return

    df = load()

    # Column names
    resp_col = 'CO2_Concentration (ppm)'
    p1 = 'Precipitation (mm)'
    p2 = 'Avg_Temp (°C)'
    p3 = 'Humidity (%)'
    p4 = 'Solar_Irradiance (W/m²)'
    p5 = 'Sea_Surface_Temp (°C)'

    # numpy arrays
    precip = df[p1].to_numpy()
    avg_temp = df[p2].to_numpy()
    humidity = df[p3].to_numpy()
    solar = df[p4].to_numpy()
    sst = df[p5].to_numpy()
    y = df[resp_col].to_numpy()

    # Multiple linear regression
    X = construct_design_matrix(precip, avg_temp, humidity, solar, sst)

    X_train, y_train, X_test, y_test = train_test_split(X, y)
    beta = normal_equations_solver(X_train, y_train)
    y_pred = predict(X_test, beta)

    se_test = squared_error(y_test, y_pred)

    # Condition number of XtX
    XtX = X_train.T @ X_train
    svals_XtX = np.linalg.svd(XtX, compute_uv=False)
    cond_X = svals_XtX[0] / svals_XtX[-1]


    print("Multiple Linear Regression Results:")
    print("Condition number cond(X):", cond_X)
    print()

    
    # Predicted vs Actual
    plotting(y_test, y_pred, "Multiple Regression: Pred vs Actual",
             returned="multiple_pred_vs_actual.png")

    
    # Polynomial regression (single predictor variable)
    poly_results = polynomial_regression_analysis(avg_temp, y, max_degree=10)

    # Error plot
    plot_errors(poly_results['train_errors'],
                poly_results['test_errors'],
                returned="poly_error_plot.png")

    
    # Condition number plot (log scale)
    plt.figure(figsize=(7,6))
    plt.semilogy(range(1, 11), poly_results['cond_numbers'], marker='o')
    plt.xlabel("Polynomial degree")
    plt.ylabel("cond(X) (log scale)")
    plt.title("Condition Number vs Polynomial Degree")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("poly_condition_plot.png")
    plt.close()

    
    # Polynomial fits
    plot_polynomial_fits(avg_temp, y, max_degree=6)

    print("Saved all plots: multiple_pred_vs_actual.png, poly_error_plot.png, poly_condition_plot.png, poly_fits.png")

if __name__ == "__main__":
    main()
