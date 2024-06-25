# main.py

import os
import warnings
import pandas as pd
import config
import random
from sklearn.exceptions import ConvergenceWarning

# Suppress warnings
warnings.filterwarnings('ignore', category=pd.errors.PerformanceWarning)
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', category=ConvergenceWarning)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# Set seed for reproducibility
import tensorflow as tf
tf.get_logger().setLevel('ERROR')
if config.global_config.get('set_seed', False):
    seed = config.global_config['set_seed']
    tf.random.set_seed(seed)
    random.seed(seed)

import data_processing as dp
from models.polynomial_interpolation import polynomial_interpolation
from models.lstm import lstm
from models.mice import mice_mlr, mice_knn
from models.autoencoder import autoencoder
from models.dlin import dlin

def run_models(data_miss):
    results = {}

    if config.run_model.get('polynomial_interpolation', False):
        predicted_data = polynomial_interpolation(data_miss.copy())
        results['polynomial_interpolation'] = predicted_data

    if config.run_model.get('lstm', False):
        predicted_data = lstm(data_miss.copy())
        results['lstm'] = predicted_data

    if config.run_model.get('mice_mlr', False):
        predicted_data = mice_mlr(data_miss.copy())
        results['mice_mlr'] = predicted_data

    if config.run_model.get('mice_knn', False):
        predicted_data = mice_knn(data_miss.copy())
        results['mice_knn'] = predicted_data
    
    if config.run_model.get('autoencoder', False):
        predicted_data = autoencoder(data_miss.copy())
        results['autoencoder'] = predicted_data
    
    if config.run_model.get('dlin', False):
        predicted_data = dlin(data_miss.copy())
        results['dlin'] = predicted_data

    return results

# Load original data and generate missing values
original_data = dp.load_and_standardize_data()
data_miss, missing_indexes = dp.generate_misses(original_data)

if config.global_config.get('allow_run_models', False):
    predictions = run_models(data_miss)
    for model_name, predicted_data in predictions.items():
        print(f"\nEvaluating model: {model_name}")
        dp.evaluate_model(original_data, predicted_data, missing_indexes)



