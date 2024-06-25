import config
from missing_generator import produce_missings
import pandas as pd
import numpy as np
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.preprocessing import StandardScaler
import hashlib
import os

def load_and_standardize_data():
    data = pd.read_csv(config.global_config['data_file'], delimiter=';')
    if 'date' in data.columns:
        data = data.drop(columns=['date'])
    scaler = StandardScaler()
    standardized_data = scaler.fit_transform(data)
    standardized_df = pd.DataFrame(standardized_data, columns=data.columns)
    return standardized_df

def ensure_dir(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)

def generate_seed(column_name):
    """Generate a seed based on the column name."""
    return int(hashlib.sha256(column_name.encode('utf-8')).hexdigest(), 16) % (10 ** 9) + 2

def generate_misses(data, 
                    perc_miss=config.global_config.get('missing_percentage', 0.1), 
                    length_miss=config.global_config.get('burst_length', 3)):


    data_miss = data.copy()
    missing_indexes = {}

    for column in data.columns:
        time_series = data[column]
        if config.global_config.get('set_seed', False):
            seed = generate_seed(column)
        else:
            seed = None
        indexes_missing, incomplete_time_series = produce_missings(time_series, perc_miss, int(length_miss), seed=seed)
        data_miss[column] = incomplete_time_series
        missing_indexes[column] = sorted(indexes_missing)

    return data_miss, missing_indexes


def find_missing_indexes(data):
    missing_indexes = {}
    for column in data.columns:
        missing_indexes[column] = data[column][data[column].isna()].index.tolist()
    return missing_indexes

def get_values_from_indexes(data, indexes_missing):
    """ 
    Given a dataset and a dictionary of indexes, return the values at those indexes.
    
    INPUT:
        - data: pandas DataFrame
        - indexes_missing: dictionary with column names as keys and lists of missing indexes as values
        
    OUTPUT:
        - missing_values: dictionary with column names as keys and lists of values at the missing indexes as values
    """
    values = {}
    
    for column, indexes in indexes_missing.items():
        values[column] = data.iloc[indexes][column].values.tolist()
    
    return values

def evaluate_model(original_data, filled_data, missing_indexes):
    real_values = get_values_from_indexes(original_data, missing_indexes)
    predicted_values = get_values_from_indexes(filled_data, missing_indexes)

    r2_scores = {}
    rmse_scores = {}
    total_real_vals = []
    total_predicted_vals = []

    for column in real_values.keys():
        real_vals = real_values[column]
        predicted_vals = predicted_values[column]

        # Calculate R-squared and RMSE
        r2 = r2_score(real_vals, predicted_vals)
        rmse = np.sqrt(mean_squared_error(real_vals, predicted_vals))

        r2_scores[column] = r2
        rmse_scores[column] = rmse
        total_real_vals.extend(real_vals)
        total_predicted_vals.extend(predicted_vals)

        # print_real_vs_predicted_values(column, real_vals, predicted_vals, missing_indexes)

    # Calculate total R-squared and RMSE
    total_r2 = r2_score(total_real_vals, total_predicted_vals)
    total_rmse = np.sqrt(mean_squared_error(total_real_vals, total_predicted_vals))

    print_evaluation_results(r2_scores, rmse_scores, total_r2, total_rmse)

    return r2_scores, rmse_scores, total_r2, total_rmse

def compare_missing_indexes(indexes1, indexes2):
    for column in indexes1.keys():
        if column not in indexes2:
            print(f"Column {column} is missing in the second set of indexes.")
            return False
        if not np.array_equal(indexes1[column], indexes2[column]):
            print(f"Indexes for column {column} are different.")
            return False
    return True

def print_evaluation_results(r2_scores, rmse_scores, total_r2, total_rmse):
    # print("R-squared scores for each column:")
    # for column, r2 in r2_scores.items():
    #     print(f"{column}: {r2:.4f}")
    # print("\nRMSE scores for each column:")
    # for column, rmse in rmse_scores.items():
    #     print(f"{column}: {rmse:.4f}")
    
    # Print total R-squared and RMSE
    print("Total R-squared: {:.4f}".format(total_r2))
    # print("Total RMSE: {:.4f}".format(total_rmse))

def print_real_vs_predicted_values(column, real_vals, predicted_vals, missing_indexes):
    print(f"\nColumn: {column}")
    print(f"{'Index':<10}{'Real Value':<20}{'Predicted Value':<20}")
    print("-" * 50)
    for index, (real, predicted) in enumerate(zip(real_vals, predicted_vals)):
        print(f"{missing_indexes[column][index]:<10}{real:<20}{predicted:<20}")
    