import pandas as pd
import numpy as np
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer, KNNImputer
from sklearn.linear_model import LinearRegression
import config

def create_mlr_imputer(max_iter=None, random_state=None):
    # Retrieve MLR config from config.py
    mice_mlr_config = config.mice_config['mlr']['base']
    max_iter = max_iter if max_iter is not None else mice_mlr_config['max_iter']
    random_state = random_state if random_state is not None else mice_mlr_config['random_state']
    imputer = IterativeImputer(estimator=LinearRegression(), max_iter=int(max_iter), random_state=random_state)
    return imputer

def create_knn_imputer(n_neighbors=None, weights=None, metric=None):
    # Retrieve KNN config from config.py
    knn_config = config.mice_config['knn']['base']
    n_neighbors = n_neighbors if n_neighbors is not None else knn_config['n_neighbors']
    weights = weights if weights is not None else knn_config['weights']
    metric = metric if metric is not None else knn_config['metric']
    imputer = KNNImputer(n_neighbors=int(n_neighbors), weights=weights, metric=metric)
    return imputer

def mice_mlr(data_miss, max_iter=None, random_state=None):
    # Use default values from config.py if no values are passed
    filled_data = data_miss.copy()
    imputer = create_mlr_imputer(max_iter=max_iter, random_state=random_state)
    imputed_data_array = imputer.fit_transform(filled_data)
    filled_data = pd.DataFrame(imputed_data_array, columns=filled_data.columns)
    
    return filled_data

def mice_knn(data_miss, n_neighbors=None, weights=None, metric=None):
    # Use default values from config.py if no values are passed
    filled_data = data_miss.copy()
    imputer = create_knn_imputer(n_neighbors=n_neighbors, weights=weights, metric=metric)
    imputed_data_array = imputer.fit_transform(filled_data)
    filled_data = pd.DataFrame(imputed_data_array, columns=filled_data.columns)
    
    return filled_data
