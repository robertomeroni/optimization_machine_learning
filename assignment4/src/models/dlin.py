import numpy as np
import pandas as pd
from sklearn.impute import IterativeImputer
from sklearn.linear_model import LinearRegression


def global_estimator(data):
    """
    Global estimator using MICE with MLR.
    
    Parameters:
    - data: pd.DataFrame, the time series matrix with missing values.
    
    Returns:
    - filled_data: data matrix filled with predicted values.
    """
    imputer = IterativeImputer(estimator=LinearRegression(), random_state=25)
    imputed_data_array = imputer.fit_transform(data)
    return pd.DataFrame(imputed_data_array, columns=data.columns)

def local_estimator(data, idx, window_size=5):
    """
    Local estimator using linear regression.
    
    Parameters:
    - data: pd.Series, the time series with missing values.
    - idx: int, the index of the missing value to predict.
    - window_size: int, the number of recent points to use for the local model.
    
    Returns:
    - local_prediction: predicted value for the missing index.
    """
    if idx < window_size:
        return np.nan  # Not enough data for local model

    local_x = np.arange(window_size).reshape(-1, 1)
    local_y = data.iloc[idx-window_size:idx].values
    
    if np.isnan(local_y).any():  # Ensure no NaNs in local_y
        return np.nan
    
    local_model = LinearRegression()
    local_model.fit(local_x, local_y)
    return local_model.predict(np.array([[window_size]]))[0]

def combine_predictions(local_prediction, global_prediction, combined_model):
    """
    Combine local and global predictions using a combined model.
    
    Parameters:
    - local_prediction: float, the local prediction.
    - global_prediction: float, the global prediction.
    - combined_model: LinearRegression, the combined regression model.
    
    Returns:
    - value: float, the combined prediction value.
    """
    return combined_model.predict([[local_prediction, global_prediction]])[0]

def fit_combined_model(data, column, valid_indices, window_size=5, batch_size=64):
    """
    Fit the combined model using local and global predictions.
    
    Parameters:
    - data: pd.DataFrame, the original data matrix.
    - column: str, the column to fit the model on.
    - valid_indices: list, indices of valid (non-missing) values.
    - window_size: int, the number of recent points to use for the local model.
    - batch_size: int, the size of batches to process for global estimation.
    
    Returns:
    - combined_model: LinearRegression, the combined regression model.
    """
    local_predictions = []
    global_predictions = []
    valid_values = []
    data_train = data.copy()

    for batch_start in range(0, len(valid_indices), batch_size):
        batch_indices = valid_indices[batch_start:batch_start + batch_size]

        # Set the batch indices to NaN in the training data
        data_train.loc[batch_indices, column] = np.nan

        # Get global predictions for the batch
        train_global_prediction = global_estimator(data_train)
        global_batch_predictions = train_global_prediction.loc[batch_indices, column]

        # Reset the batch indices to their original values
        data_train.loc[batch_indices, column] = data.loc[batch_indices, column]

        for idx in batch_indices:
            train_local_prediction = local_estimator(data[column], idx, window_size)

            local_predictions.append(train_local_prediction if not np.isnan(train_local_prediction) else global_batch_predictions.loc[idx])
            global_predictions.append(global_batch_predictions.loc[idx])
            valid_values.append(data[column].loc[idx])

    combined_model = LinearRegression()
    X = np.column_stack((local_predictions, global_predictions))
    combined_model.fit(X, valid_values)

    return combined_model

def dlin(data, window_size=1, batch_size=128):
    """
    Impute missing values in a time series matrix using the DLin technique.
    
    Parameters:
    - data: pd.DataFrame, the matrix with missing values.
    - window_size: int, the number of recent points to use for the local model.
    - batch_size: int, the size of batches to process for global estimation.
    
    Returns:
    - data_imputed: pd.DataFrame, the matrix with missing values imputed.
    """
    window_size = int(window_size)
    batch_size = int(batch_size)
    data_imputed = data.copy()
    mice_filled_data = global_estimator(data)

    for column in data.columns:
        col_data = data[column]
        missing_indices = col_data[col_data.isna()].index
        valid_indices = col_data[col_data.notna()].index

        combined_model = fit_combined_model(data, column, valid_indices, window_size, batch_size)

        for idx in missing_indices:
            local_prediction = local_estimator(col_data, idx, window_size)
            global_prediction = mice_filled_data.loc[idx, column]

            data_imputed.loc[idx, column] = combine_predictions(local_prediction if not np.isnan(local_prediction) else global_prediction, global_prediction, combined_model)

    return data_imputed

