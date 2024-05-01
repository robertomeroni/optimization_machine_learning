import numpy as np
import data_processing as dp
from sklearn.covariance import GraphicalLasso
from data_paths import location_data_path
import pandas as pd
import cvxpy as cp

def glasso_model(data, alpha=0.001, T=0):
    """
    Apply Graphical Lasso to estimate and print the precision matrix with variable names.
    :param data: DataFrame with observations as rows and features as columns.
    :param alpha: The regularization parameter controlling the sparsity.
    :param T: Threshold value to control the sparsity of the graph. Elements with absolute value <= T are set to zero.
    :return: Precision matrix as a DataFrame with variable names, threshold applied.
    """
    
    # Ensure the data is a DataFrame and extract the feature names
    if not isinstance(data, pd.DataFrame):
        raise ValueError("Data must be a pandas DataFrame.")

    # Create and fit the Graphical Lasso model
    model = GraphicalLasso(alpha=alpha, max_iter=10000, tol=1e-3)
    model.fit(data)

    # Retrieve the precision matrix and apply the threshold
    precision_matrix = model.precision_
    precision_matrix[np.abs(precision_matrix) <= T] = 0

    return precision_matrix


def glasso_cvxpy(data, alpha=0.01, T=0):
    """
    Graphical Lasso implementation using CVXPY.
    
    Args:
    data (pd.DataFrame): The input data matrix (observations x features).
    alpha (float): Regularization parameter.

    Returns:
    pd.DataFrame: Estimated precision matrix with variable names.
    """
    if not isinstance(data, pd.DataFrame):
        raise ValueError("Data must be a pandas DataFrame.")
    
    # Compute the empirical covariance matrix
    S = np.cov(data.values, rowvar=False)
    
    # Number of variables (features)
    p = S.shape[0]
    
    # Define the variable for the precision matrix
    Theta = cp.Variable((p, p), symmetric=True)
    
    # The objective function
    log_det_Theta = cp.log_det(Theta)
    trace_STheta = cp.trace(cp.matmul(S, Theta))
    l1_penalty = alpha * cp.sum(cp.abs(Theta - cp.diag(cp.diag(Theta))))
    
    # The problem
    problem = cp.Problem(cp.Minimize(trace_STheta - log_det_Theta + l1_penalty))
    
    # Solve the problem
    problem.solve()

    # Retrieve the precision matrix and apply the threshold
    precision_matrix = Theta.value
    precision_matrix[np.abs(precision_matrix) <= T] = 0

    return pd.DataFrame(precision_matrix, columns=data.columns, index=data.columns)
