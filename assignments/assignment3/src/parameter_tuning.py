import data_processing as dp
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from typing import List, Dict, Any
from itertools import product
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.base import clone
from sklearn.model_selection import cross_val_score
from model_config import linspace_plot_config, global_config
import json
import os

def get_parameters_from_results_file(results_file):
    best_model_info = find_best_models_from_file(results_file)
    return best_model_info['params']

def get_r2_from_results_file(results_file):
    best_model_info = find_best_models_from_file(results_file)
    return best_model_info['r2']

def get_r2_from_fss_file(fss_file):
    if not os.path.exists(fss_file):
        print(f"FSS file {fss_file} does not exist.")
        return None
    
    with open(fss_file, 'r') as f:
        data = json.load(f)
    return max(data['scores'])

def get_r2_from_shuffle_file(shuffle_file):
    if not os.path.exists(shuffle_file):
        print(f"Shuffle file {shuffle_file} does not exist.")
        return None
    
    with open(shuffle_file, 'r') as f:
        data = json.load(f)
    return data['mean_shuffled_score']

def get_all_r2_scores():
    r2_scores = {}

    results_dir = 'outputs'
    fss_dir = os.path.join(results_dir, 'fss')
    shuffle_dir = os.path.join(results_dir, 'shuffle')

    for model_name, config in linspace_plot_config.items():
        # Define file paths
        results_file = os.path.join(results_dir, f"{model_name.replace(' ', '_')}_results.json")
        fss_file = os.path.join(fss_dir, f"fss_results_{model_name.replace(' ', '_')}.json")
        shuffle_file = os.path.join(shuffle_dir, f"shuffle_results_{model_name.replace(' ', '_')}.json")

        if not os.path.exists(results_file):
            print(f"Results file {results_file} not found for {model_name}")
            continue
        
        # Get R² scores from results, FSS, and shuffle files
        parameters = get_parameters_from_results_file(results_file)
        test_r2 = get_r2_from_results_file(results_file)
        fss_r2 = get_r2_from_fss_file(fss_file)
        shuffle_r2 = get_r2_from_shuffle_file(shuffle_file)
        
        r2_scores[model_name] = {
            "test_r2": test_r2,
            "fss_r2": fss_r2,
            "shuffle_r2": shuffle_r2
        }

        print(f"Model: {model_name}")
        print(f"Parameters: {parameters}")
        print(f"Best R²: {test_r2}")
        print(f"FSS R²: {fss_r2}")
        print(f"Shuffle Average R²: {shuffle_r2}")
        print()

    return r2_scores

def forward_subset_selection(X, y, model, max_features=None, scoring='r2'):
    selected_features = []
    remaining_features = list(X.columns)
    best_score = -np.inf
    scores = []

    if max_features is None:
        max_features = len(remaining_features)

    while len(selected_features) < max_features and remaining_features:
        scores_with_candidates = []
        
        for feature in remaining_features:
            candidates = selected_features + [feature]
            X_subset = X[candidates]
            model_clone = clone(model)
            score = np.mean(cross_val_score(model_clone, X_subset, y, cv=5, scoring=scoring))
            scores_with_candidates.append((score, feature))
        
        scores_with_candidates.sort(reverse=True)
        best_new_score, best_new_feature = scores_with_candidates[0]

        if best_new_score > best_score:
            best_score = best_new_score
            selected_features.append(best_new_feature)
            remaining_features.remove(best_new_feature)
            scores.append(best_score)
        else:
            break
    
    return selected_features, scores


def from_best_model_to_fss(model_name: str, file_path: str, X, y):
    best_model_info = find_best_models_from_file(file_path)
    if not best_model_info:
        print("No valid model information found.")
        return [], []
    
    best_model = best_model_info['model']
    best_params = best_model_info['params']
    
    model = clone(best_model).set_params(**best_params)
    
    # Scale features
    scaler = StandardScaler()
    X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)
    
    selected_features, scores = forward_subset_selection(X_scaled, y, model)
    
    # print("Selected Features:", selected_features)
    # print("Scores for each step:", scores)
    
    fss_results = {
        "selected_features": selected_features,
        "scores": scores
    }

    if global_config.get('save_fss_data', False):
        save_folder = 'outputs/fss'
        os.makedirs(save_folder, exist_ok=True)
        save_file_path = os.path.join(save_folder, f'fss_results_{model_name.replace(' ', '_')}.json')
        with open(save_file_path, 'w') as f:
            json.dump(fss_results, f, indent=4)
        print(f"FSS results saved to {save_file_path}")

    return selected_features, scores



def apply_gridsearchCV(X, y, estimator, param_grid): 
    print("Applying GridSearch on training set...")
    print()
    for scoring in ['neg_mean_squared_error', 'r2']:   
        grid_search = GridSearchCV(estimator=estimator, param_grid=param_grid, cv=5, scoring=scoring)
        grid_search.fit(X, y)
        best_params = grid_search.best_params_
        best_model = grid_search.best_estimator_
        best_score = grid_search.best_score_
        print("Scoring:", scoring)
        print("Best Parameters:", best_params)
        print("Best Model:", best_model)
        print("Best Score:", best_score)
        print("\n")

def parameter_tuning(
    model_name: str, 
    estimator, 
    param_grid, 
    X_train, 
    X_test, 
    y_train, 
    y_test, 
    scaler_y, 
    apply_gridsearch: bool = False, 
    print_results: bool = False,
    print_best: bool = False,
    output_file: str = None,  
    allow_file_write: bool = False
):
    results = []

    if print_results or apply_gridsearch:
        print("-" * 40)
        print(f"{model_name}:")
        print()

    if apply_gridsearch:
        apply_gridsearchCV(X_train, y_train, estimator, param_grid)

    keys, values = zip(*param_grid.items())
    combinations = [dict(zip(keys, v)) for v in product(*values)]

    for params in combinations:
        model = estimator.set_params(**params)
        model.fit(X_train, y_train)
        
        y_pred = model.predict(X_test)
        y_pred = scaler_y.inverse_transform(y_pred.reshape(-1, 1)).flatten()
        
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        
        # Convert numpy types to regular Python types
        params = {k: (int(v) if isinstance(v, np.integer) else float(v) if isinstance(v, np.floating) else v) for k, v in params.items()}
        
        result = {
            'model': model_name,
            'params': params,
            'mse': mse,
            'r2': r2
        }
        results.append(result)
        
        if print_results:
            print(f"Parameters: {params}")
            print(f"Mean Squared Error (MSE): {mse}")
            print(f"R-squared (R2): {r2}")
            print()

    if print_best:
        find_best_models(results)

    if allow_file_write and output_file:
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=4)
    
    if print_results or apply_gridsearch:
        print("-" * 40)
        print()

    return results


def find_best_models_from_file(file_path: str) -> Dict[str, Dict[str, Any]]:
    if not os.path.exists(file_path):
        print(f"File {file_path} does not exist. Skipping best model search.")
        return {}

    with open(file_path, 'r') as f:
        results = json.load(f)
    return find_best_models(results)

def find_best_models(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    best_r2 = {'r2': float('-inf'), 'params': {}, 'model': None}
    
    for result in results:
        if result['r2'] > best_r2['r2']:
            best_r2 = result
    
    # Find the actual estimator object
    model_name = best_r2['model']
    estimator = linspace_plot_config[model_name]['estimator']
    best_r2['model'] = estimator

    if global_config.get('print_best', False):
        print("Best Model by R-squared:")
        print(f"Model: {model_name}")
        print(f"Parameters: {best_r2['params']}")
        print(f"R-squared: {best_r2['r2']}")
        print(f"MSE: {best_r2['mse']}\n")
    
    return best_r2


