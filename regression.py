import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# Loading the data from the .csv file into a spreadsheet of values
def load():
    data = pd.read_csv("climate_change_dataset.csv")
    return data


# Train / test split
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






# Polynomial design matrix (for one predictor)
def construct_poly_design_matrix(x, degree):
    """
    Build polynomial design matrix [1, x, x^2, ..., x^degree] for a single predictor x.
    x may be 1D numpy array or pandas Series.
    Returns numpy array shape (n, degree+1).
    """
    x_arr = np.asarray(x).reshape(-1)
    n = len(x_arr)
    X = np.ones((n, degree + 1))
    for d in range(1, degree + 1):
        X[:, d] = x_arr ** d
    return X



# Solving normal equations
def normal_equations_solver(X, y, cond_thresh=1e12):
    # Solve (X^T X) beta = X^T y

    XtX = X.T @ X
    Xty = X.T @ y
    
    beta = np.linalg.solve(XtX, Xty)
    info = {'Method used': 'normal equations'}
    return beta, info



def qr_solver(X, y):
    """
    Solve least-squares using QR: X = Q R, then solve R beta = Q^T y.
    Returns beta and info dict.
    """
    X = np.asarray(X)
    y = np.asarray(y).reshape(-1)
    Q, R = np.linalg.qr(X, mode='reduced')
    beta = np.linalg.solve(R, Q.T @ y)
    info = {'Method used': 'QR factorization'}
    return beta, info



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


# -----------------------------
# Plotting helpers
# -----------------------------
def plotting(y_actual, y_hat, title='Predicted vs Actual', out_path='pred_vs_actual.png'):
    y_actual = np.asarray(y_actual).reshape(-1)
    y_hat = np.asarray(y_hat).reshape(-1)
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
    plt.savefig(out_path)
    plt.close()
    return out_path


def plot_errors(train_errs, test_errs, out_path='error_plot.png'):
    degrees = np.arange(1, len(train_errs) + 1)
    plt.figure(figsize=(7,5))
    plt.plot(degrees, train_errs, marker='o', label='Training Error (SE)')
    plt.plot(degrees, test_errs, marker='o', label='Test Error (SE)')
    plt.xlabel('Polynomial Degree')
    plt.ylabel('Squared Error')
    plt.title('Training vs Test Error as Degree Increases')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()
    return out_path

def plot_polynomial_fits(x, y, max_degree=5, out_path='poly_fits.png', solver=qr_solver):
    """
    Plot data points and fitted polynomials of degrees 1..max_degree.
    x, y are 1D arrays (single predictor).
    solver: function that takes (X, y) and returns (beta, info)
    """
    x = np.asarray(x).reshape(-1)
    y = np.asarray(y).reshape(-1)
    plt.figure(figsize=(8,6))
    plt.scatter(x, y, color='gray', s=15, label='Data')
    x_grid = np.linspace(x.min(), x.max(), 500)
    for deg in range(1, max_degree + 1):
        X_train = construct_poly_design_matrix(x, deg)
        beta, _ = solver(X_train, y)
        X_grid = construct_poly_design_matrix(x_grid, deg)
        y_grid = predict(X_grid, beta)
        plt.plot(x_grid, y_grid, label=f"deg {deg}")
    plt.xlabel('Predictor (x)')
    plt.ylabel('Response (y)')
    plt.title('Polynomial fits of different degrees')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()
    return out_path

# -----------------------------
# Polynomial regression analysis (degree sweep)
# -----------------------------
def polynomial_regression_analysis(x, y, max_degree=10, solver=qr_solver, split_ratio=0.8, seed=None):
    """
    For degrees 1..max_degree:
      - build polynomial design matrices
      - fit on training data
      - compute training & test squared error
      - compute singular values & condition number of X^T X on training set
    Returns dict with lists: train_errors, test_errors, cond_numbers, singular_values_list
    """
    # Prepare train/test splits (use the single predictor x for splitting so we can preserve pairing)
    x_arr = np.asarray(x).reshape(-1)
    y_arr = np.asarray(y).reshape(-1)
    X_all = x_arr  # for clarity
    X_tr, y_tr, X_te, y_te = train_test_split(X_all, y_arr, split_ratio=split_ratio, seed=seed)
    train_errors = []
    test_errors = []
    cond_numbers = []
    svals_list = []
    for deg in range(1, max_degree + 1):
        X_train = construct_poly_design_matrix(X_tr, deg)
        X_test = construct_poly_design_matrix(X_te, deg)
        beta, info = solver(X_train, y_tr)
        y_tr_hat = predict(X_train, beta)
        y_te_hat = predict(X_test, beta)
        train_errors.append(squared_error(y_tr, y_tr_hat))
        test_errors.append(squared_error(y_te, y_te_hat))
        svals = np.linalg.svd(X_train.T @ X_train, compute_uv=False)
        svals_list.append(svals)
        cond_numbers.append(condition_number(svals))
    return {
        'train_errors': train_errors,
        'test_errors': test_errors,
        'cond_numbers': cond_numbers,
        'svals_list': svals_list
    }

# -----------------------------
# Example main that ties everything together for your dataset
# -----------------------------
def main_demo():
    # Check CSV exists
    csv_path = Path("climate_change_dataset.csv")
    if not csv_path.exists():
        print("Put 'climate_change_dataset.csv' in the working directory and re-run.")
        return

    # load
    df = load()
    print("Loaded data shape:", df.shape)
    # adjust these column names if your CSV uses slightly different names
    resp_col = 'CO2_Concentration (ppm)'
    p1 = 'Precipitation (mm)'
    p2 = 'Avg_Temp (°C)'
    p3 = 'Humidity (%)'
    p4 = 'Solar_Irradiance (W/m²)'
    p5 = 'Sea_Surface_Temp (°C)'

    # ensure columns exist
    for col in [resp_col, p1, p2, p3, p4, p5]:
        if col not in df.columns:
            raise ValueError(f"Expected column '{col}' not found in CSV. Columns available: {list(df.columns)}")

    # extract arrays
    precip = df[p1].to_numpy()
    avg_temp = df[p2].to_numpy()
    humidity = df[p3].to_numpy()
    solar = df[p4].to_numpy()
    sst = df[p5].to_numpy()
    y = df[resp_col].to_numpy()

    # construct multiple-predictor design matrix and run a single linear fit
    X = construct_design_matrix(precip, avg_temp, humidity, solar, sst)
    X_train, y_train, X_test, y_test = train_test_split(X, y, split_ratio=0.8, seed=42)
    beta_ne, info_ne = normal_equations_solver(X_train, y_train)
    y_test_pred_ne = predict(X_test, beta_ne)
    se_test_ne = squared_error(y_test, y_test_pred_ne)
    svals_XtX = singular_values_XtX(X_train)
    cond_XtX = condition_number(svals_XtX)
    print("Multiple linear regression (normal equations):")
    print("method info:", info_ne)
    print("Test squared error:", se_test_ne)
    print("Condition number (X^T X):", cond_XtX)

    # Plot predicted vs actual (multiple-regression)
    plotting(y_test, y_test_pred_ne, title='Multiple reg: Pred vs Actual', out_path='multiple_pred_vs_actual.png')
    print("Saved multiple_pred_vs_actual.png")

    # Now polynomial regression analysis using avg_temp as the single predictor
    poly_results = polynomial_regression_analysis(avg_temp, y, max_degree=10, solver=qr_solver, split_ratio=0.8, seed=42)
    plot_errors(poly_results['train_errors'], poly_results['test_errors'], out_path='poly_error_plot.png')
    print("Saved poly_error_plot.png")
    # save condition numbers plot
    plt.figure(figsize=(7,5))
    plt.plot(np.arange(1, len(poly_results['cond_numbers'])+1), poly_results['cond_numbers'], marker='o')
    plt.xlabel('Polynomial degree')
    plt.ylabel('Condition number of X^T X (train)')
    plt.title('Condition number vs polynomial degree')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('poly_condition_plot.png')
    plt.close()
    print("Saved poly_condition_plot.png")

    # plot polynomial fits (visual)
    plot_polynomial_fits(avg_temp, y, max_degree=6, out_path='poly_fits.png', solver=qr_solver)
    print("Saved poly_fits.png")

if __name__ == "__main__":
    main_demo()

