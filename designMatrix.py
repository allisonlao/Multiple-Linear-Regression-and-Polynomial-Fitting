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
    ones = np.ones((len(precip), 1))

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
    """
    Solve (X^T X) beta = X^T y if well-conditioned; otherwise fall back to pseudoinverse.
    Returns (beta, info_dict).
    """
    XtX = X.T @ X
    Xty = X.T @ y
    s = np.linalg.svd(XtX, compute_uv=False)
    cond = np.nan
    if len(s) > 0 and s[-1] > 0:
        cond = s[0] / s[-1]
    info = {'cond_XtX': cond}
    try:
        if cond < cond_thresh:
            beta = np.linalg.solve(XtX, Xty)
            info['method'] = 'solve'
        else:
            beta = np.linalg.pinv(X) @ y
            info['method'] = 'pinv_due_to_condition'
    except np.linalg.LinAlgError:
        beta = np.linalg.pinv(X) @ y
        info['method'] = 'pinv_exception'
    return beta, info

def qr_solver(X, y):
    """
    Solve via QR: X = Q R, beta = R^{-1} Q^T y
    """
    Q, R = np.linalg.qr(X, mode='reduced')
    beta = np.linalg.solve(R, Q.T @ y)
    return beta, {'method': 'qr'}

# -----------------------------
# Predict & evaluate
# -----------------------------
def predict(X, beta):
    return X @ beta

def squared_error(y_true, y_pred):
    return np.sum((y_true - y_pred) ** 2)

def rmse(y_true, y_pred):
    return np.sqrt(np.mean((y_true - y_pred) ** 2))

# -----------------------------
# SVD & condition number
# -----------------------------
def singular_values_XtX(X):
    XtX = X.T @ X
    s = np.linalg.svd(XtX, compute_uv=False)
    return s

def condition_number_from_svals(svals):
    if len(svals) == 0:
        return np.nan
    if svals[-1] == 0:
        return np.inf
    return float(svals[0] / svals[-1])

# -----------------------------
# Simple plotting
# -----------------------------
def plot_pred_vs_actual(y_true, y_pred, title='Predicted vs Actual', out_path='pred_vs_actual.png'):
    plt.figure(figsize=(6,6))
    plt.scatter(y_true, y_pred)
    mn = min(min(y_true), min(y_pred))
    mx = max(max(y_true), max(y_pred))
    plt.plot([mn, mx], [mn, mx])
    plt.xlabel('Actual')
    plt.ylabel('Predicted')
    plt.title(title)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()
    return out_path

# -----------------------------
# Demo pipeline (example usage)
# -----------------------------
def demo_pipeline(csv_path,
                  response_col,
                  predictor_cols,
                  poly_features=None,
                  interaction_pairs=None,
                  standardize=True,
                  ratio=0.8,
                  seed=42,
                  shuffle=True):
    # load and prepare
    X, y_all, data = load(csv_path, response_col, predictor_cols, fill_method='mean')
    print("Data sample:")
    print(data.head())

    # split
    X_train, X_test, y_train, y_test = train_test_split(X, y_all, ratio=ratio, seed=seed, shuffle=shuffle)

    # design matrices
    X_train, feature_names = construct_design_matrix(X_train, add_intercept=True, poly_features=poly_features, interaction_pairs=interaction_pairs, standardize=standardize)
    X_test, _ = construct_design_matrix(X_test, add_intercept=True, poly_features=poly_features, interaction_pairs=interaction_pairs, standardize=standardize)

    print("Design matrix features:", feature_names)
    print("Shapes:", X_train.shape, X_test.shape)

    # Normal equations
    beta_ne, info_ne = normal_equations_solver(X_train, y_train)
    y_train_pred_ne = predict(X_train, beta_ne)
    y_test_pred_ne = predict(X_test, beta_ne)
    train_err_ne = squared_error(y_train, y_train_pred_ne)
    test_err_ne = squared_error(y_test, y_test_pred_ne)
    svals = singular_values_XtX(X_train)
    cond = condition_number_from_svals(svals)

    print("Normal eq info:", info_ne)
    print(f"Train error (NE): {train_err_ne:.6f}, Test error (NE): {test_err_ne:.6f}")
    print(f"Condition number of X^T X (train): {cond:.4e}")

    # QR solver
    beta_qr, info_qr = qr_solver(X_train, y_train)
    y_test_pred_qr = predict(X_test, beta_qr)
    test_err_qr = squared_error(y_test, y_test_pred_qr)
    print("QR info:", info_qr)
    print(f"Test error (QR): {test_err_qr:.6f}")

    # Plot predicted vs actual on test set (NE)
    out_path = plot_pred_vs_actual(y_test, y_test_pred_ne, title='NE: Predicted vs Actual (test)', out_path='pred_vs_actual_ne.png')
    print("Saved plot to", out_path)

    return {
        'X_train': X_train, 'X_test': X_test,
        'y_train': y_train, 'y_test': y_test,
        'beta_ne': beta_ne, 'beta_qr': beta_qr,
        'train_err_ne': train_err_ne, 'test_err_ne': test_err_ne,
        'test_err_qr': test_err_qr,
        'svals_XtX': svals, 'cond_XtX': cond,
        'feature_names': feature_names,
        'plot_path': out_path
    }

# -----------------------------
# Example of running demo_pipeline
# -----------------------------
if __name__ == "__main__":
    csv_path = 'climate_change_dataset.csv'   # put your CSV in same directory or give full path
    if Path(csv_path).exists():
        response_col = 'CO2_Concentration (ppm)'
        predictor_cols = ['Precipitation (mm)', 'Avg_Temp (°C)', 'Humidity (%)', 'Solar_Irradiance (W/m²)', 'Sea_Surface_Temp (°C)']
        # Example: add interaction Temp * Humidity and standardize features
        results = demo_pipeline(csv_path, response_col, predictor_cols,
                                poly_features=None,
                                interaction_pairs=[('Avg_Temp (°C)', 'Humidity (%)')],
                                standardize=True,
                                ratio=0.8, seed=42, shuffle=True)
        print("Top singular values of X^T X:", results['svals_XtX'][:6])
    else:
        print(f"Place your dataset named '{csv_path}' in this directory and run this script.")
