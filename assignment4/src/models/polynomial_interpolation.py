import numpy as np
from scipy.interpolate import interp1d


def polynomial_interpolation(data_miss, order=3):
    """
    This function fills missing values in a DataFrame using polynomial interpolation.
    
    INPUT:
        - data_miss: pandas DataFrame with missing values
        - order: Order of the polynomial interpolation (default is cubic)
        
    OUTPUT:
        - filled_data: DataFrame with missing values filled using polynomial interpolation
    """
    filled_data = data_miss.copy()
    missing_indexes = {}
    
    for column in data_miss.columns:
        time_series = data_miss[column]
        # Get indexes of non-missing values
        valid_indexes = np.where(~time_series.isna())[0]
        miss_indexes = np.where(time_series.isna())[0]
        missing_indexes[column] = miss_indexes
        
        # Ensure there are enough valid points for interpolation
        if len(valid_indexes) < order + 1:
            raise ValueError(f"Not enough data points to perform {order}-order polynomial interpolation for column '{column}'.")
        
        # Perform polynomial interpolation
        f = interp1d(valid_indexes, time_series.dropna(), kind=order, fill_value="extrapolate")
        
        # Fill missing values
        filled_data.loc[miss_indexes, column] = f(miss_indexes)
        
    return filled_data
