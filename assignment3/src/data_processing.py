import parameter_tuning as pt
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import json
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import cm
from typing import Dict, Any, List
import os
from scipy.interpolate import griddata
from model_config import global_config, plot_config, model_plot_config, model_plot_config_extended, linspace_plot_config
from sklearn.tree import plot_tree
from sklearn.decomposition import PCA
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, AdaBoostRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.kernel_ridge import KernelRidge
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import ElasticNet
from sklearn.tree import DecisionTreeRegressor
from sklearn.preprocessing import LabelEncoder, PolynomialFeatures
from sklearn.utils import shuffle
from sklearn.base import clone
from sklearn.model_selection import cross_val_score
from sklearn.metrics import r2_score
import scipy.stats as stats
import json
import os
from matplotlib.ticker import LinearLocator

def get_polynomial_features(X_train_scaled, X_test_scaled, degree=2):
    poly = PolynomialFeatures(degree=degree)
    X_train_poly = poly.fit_transform(X_train_scaled)
    X_test_poly = poly.transform(X_test_scaled)
    return X_train_poly, X_test_poly

def plot_actual_vs_predicted(model_name: str, file_path: str, X_train, X_test, y_train, y_test, scaler_y):
    # Load the best model's parameters
    best_model_info = pt.find_best_models_from_file(file_path)
    if not best_model_info:
        print(f"No valid model information found for {model_name}.")
        return
    
    best_model = best_model_info['model']
    best_params = best_model_info['params']
    
    # Clone the best model and set its parameters
    model = clone(best_model).set_params(**best_params)
    
    # Fit the model on the training data
    model.fit(X_train, y_train)
    
    # Predict on the test data
    y_pred_scaled = model.predict(X_test)
    
    # Scale back y_pred to the original scale
    y_pred = scaler_y.inverse_transform(y_pred_scaled.reshape(-1, 1)).flatten()
    
    # Plot the actual vs. predicted values
    plt.figure(figsize=(10, 6))
    plt.scatter(y_test, y_pred, alpha=0.6, color='b')
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
    plt.title(f'Actual vs Predicted Values for {model_name}')
    plt.xlabel('Actual Values')
    plt.ylabel('Predicted Values')
    plt.grid(True)
    
    # Save the plot
    if global_config.get('save_actual_vs_predicted_plot', False):
        save_folder = 'fig/actual_vs_predicted'
        os.makedirs(save_folder, exist_ok=True)
        plot_filename = os.path.join(save_folder, f'actual_vs_predicted_{model_name.replace(" ", "_")}.png')
        plt.savefig(plot_filename)
        print(f'Actual vs Predicted plot saved to {plot_filename}')

    plt.show()
    plt.close()

def plot_model_errors(model_name: str, file_path: str, X_train, X_test, y_train, y_test, scaler_y):
    # Load the best model's parameters
    best_model_info = pt.find_best_models_from_file(file_path)
    if not best_model_info:
        print(f"No valid model information found for {model_name}.")
        return
    
    best_model = best_model_info['model']
    best_params = best_model_info['params']
    
    # Clone the best model and set its parameters
    model = clone(best_model).set_params(**best_params)
    
    # Fit the model on the training data
    model.fit(X_train, y_train)
    
    # Predict on the test data
    y_pred_scaled = model.predict(X_test)
    
    # Scale back y_pred to the original scale
    y_pred = scaler_y.inverse_transform(y_pred_scaled.reshape(-1, 1)).flatten()
    
    # Calculate errors (residuals)
    errors = y_test - y_pred
    
    # Plot the errors
    plt.figure(figsize=(10, 6))
    plt.scatter(range(len(errors)), errors, alpha=0.6, color='b')
    plt.axhline(y=0, color='r', linestyle='-')
    plt.title(f'Errors of Best Model for {model_name}')
    plt.xlabel('Test Data Points')
    plt.ylabel('Error (Residuals)')
    plt.grid(True)
    
    # Save the plot
    if global_config.get('save_errors_plot', False):
        save_folder = 'fig/errors'
        os.makedirs(save_folder, exist_ok=True)
        plot_filename = os.path.join(save_folder, f'errors_{model_name.replace(" ", "_")}.png')
        plt.savefig(plot_filename)
        print(f'Error plot saved to {plot_filename}')
    
    # Show the plot
    plt.show()
    plt.close()

def plot_fss_r2_scores(model_name: str, json_file: str, save_path='fig/fss_performance'):
    # Create the directory if it doesn't exist
    os.makedirs(save_path, exist_ok=True)

    # Load FSS JSON file
    with open(json_file, 'r') as f:
        data = json.load(f)

    # Extract scores
    scores = data.get('scores', [])

    # Get the results file path from linspace_plot_config
    results_file = linspace_plot_config[model_name]['output_file']

    # Load the best R² from results file
    best_model_info = pt.find_best_models_from_file(results_file)
    best_r2 = best_model_info['r2']

    # Plot R² scores
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, len(scores) + 1), scores, marker='o', linestyle='-', color='b', label='FSS R² Scores')
    plt.ylim(0, 1)
    plt.axhline(y=best_r2, color='r', linestyle='--', label=f'Best R²: {best_r2:.4f}')
    plt.title(f'R² Scores at Each Step of FSS for {model_name}')
    plt.xlabel('Number of Features')
    plt.ylabel('R² Score')
    plt.legend()
    plt.grid(True)
    
    # Save the plot
    if global_config.get('save_fss_plot', False):
        plot_filename = os.path.join(save_path, f'fss_r2_{model_name.replace(" ", "_")}.png')
        plt.savefig(plot_filename)
        plt.show()
        plt.close()
        print(f'FSS R² plot saved to {plot_filename}')
    
    # Show the plot
    plt.show()
    plt.close()

def plot_model_performance(json_file, save_path='fig/model_performances'):
    # Create the directory if it doesn't exist
    os.makedirs(save_path, exist_ok=True)
    
    # Load JSON file
    with open(json_file, 'r') as f:
        data = json.load(f)
    
    # Extract parameters, mse, and r2
    records = []
    for entry in data:
        model = entry['model']
        params = entry['params']
        mse = entry['mse']
        r2 = entry['r2']
        # Flatten the params dictionary and combine with mse and r2
        record = {**params, 'mse': mse, 'r2': r2}
        records.append(record)
    
    # Create a DataFrame
    df = pd.DataFrame(records)
    
    # Extract parameters names (assuming there are two parameters)
    param1 = df.columns[0]
    param2 = df.columns[1]
    
    # Encode categorical parameters if necessary
    label_encoders = {}
    if df[param1].dtype == 'object':
        le = LabelEncoder()
        df[param1] = le.fit_transform(df[param1])
        label_encoders[param1] = le
    if df[param2].dtype == 'object':
        le = LabelEncoder()
        df[param2] = le.fit_transform(df[param2])
        label_encoders[param2] = le
    
    # Create meshgrid for the surface plot
    param1_values = np.unique(df[param1])
    param2_values = np.unique(df[param2])
    param1_grid, param2_grid = np.meshgrid(param1_values, param2_values)
    
    # Plot R2 in 3D surface
    r2_grid = np.array(df.pivot_table(index=param2, columns=param1, values='r2').values)
    fig_r2 = plt.figure(figsize=(12, 8))
    ax_r2 = fig_r2.add_subplot(111, projection='3d')
    surf_r2 = ax_r2.plot_surface(param1_grid, param2_grid, r2_grid, cmap='viridis', edgecolor='k', linewidth=0.5)
    fig_r2.colorbar(surf_r2, ax=ax_r2, shrink=0.5, aspect=5, pad=0.1)
    ax_r2.set_xlabel(param1)
    ax_r2.set_ylabel(param2)
    ax_r2.set_zlabel('R2')
    ax_r2.set_title(f'R2 Surface Plot for {os.path.basename(json_file)}')
    if param1 in label_encoders:
        ax_r2.set_xticks(param1_values)
        ax_r2.set_xticklabels(label_encoders[param1].inverse_transform(param1_values), rotation=45)
    if param2 in label_encoders:
        ax_r2.set_yticks(param2_values)
        ax_r2.set_yticklabels(label_encoders[param2].inverse_transform(param2_values), rotation=45)
    ax_r2.zaxis.set_major_locator(LinearLocator(10))
    ax_r2.zaxis.set_major_formatter('{x:.02f}')
    ax_r2.view_init(elev=30, azim=120)  # Adjust the view angle for better visibility
    r2_plot_filename = os.path.join(save_path, f'{os.path.splitext(os.path.basename(json_file))[0]}_r2_plot.png')
    plt.savefig(r2_plot_filename, bbox_inches='tight')
    plt.show()
    plt.close(fig_r2)
    
    # Plot MSE in 3D surface
    mse_grid = np.array(df.pivot_table(index=param2, columns=param1, values='mse').values)
    fig_mse = plt.figure(figsize=(12, 8))
    ax_mse = fig_mse.add_subplot(111, projection='3d')
    surf_mse = ax_mse.plot_surface(param1_grid, param2_grid, mse_grid, cmap='inferno', edgecolor='k', linewidth=0.5)
    fig_mse.colorbar(surf_mse, ax=ax_mse, shrink=0.5, aspect=5, pad=0.1)
    ax_mse.set_xlabel(param1)
    ax_mse.set_ylabel(param2)
    ax_mse.set_zlabel('MSE')
    ax_mse.set_title(f'MSE Surface Plot for {os.path.basename(json_file)}')
    if param1 in label_encoders:
        ax_mse.set_xticks(param1_values)
        ax_mse.set_xticklabels(label_encoders[param1].inverse_transform(param1_values), rotation=45)
    if param2 in label_encoders:
        ax_mse.set_yticks(param2_values)
        ax_mse.set_yticklabels(label_encoders[param2].inverse_transform(param2_values), rotation=45)
    ax_mse.zaxis.set_major_locator(LinearLocator(10))
    ax_mse.zaxis.set_major_formatter('{x:.02f}')
    ax_mse.view_init(elev=30, azim=120)  # Adjust the view angle for better visibility
    mse_plot_filename = os.path.join(save_path, f'{os.path.splitext(os.path.basename(json_file))[0]}_mse_plot.png')
    plt.savefig(mse_plot_filename, bbox_inches='tight')
    plt.show()
    plt.close(fig_mse)

def plot_all_models_performances(linspace_plot_config):
    for model_name, config in linspace_plot_config.items():
        if config.get('plot', False):
            json_file = config['output_file']
            if os.path.exists(json_file):
                plot_model_performance(json_file, save_path='fig/model_performances')

def generate_linspaces(config):
    linspaces = {}

    for model, settings in config.items():
        params = settings['params']
        param1_name, param1_range = list(params.items())[0]
        param2_name, param2_range = list(params.items())[1]
        
        if 'min' in param1_range and 'max' in param1_range:
            param1_min = param1_range['min']
            param1_max = param1_range['max']
            linspace1 = np.linspace(param1_min, param1_max, 10)
        elif 'options' in param1_range:
            linspace1 = param1_range['options']
        
        if 'min' in param2_range and 'max' in param2_range:
            param2_min = param2_range['min']
            param2_max = param2_range['max']
            linspace2 = np.linspace(param2_min, param2_max, 10)
        elif 'options' in param2_range:
            linspace2 = param2_range['options']
        
        linspaces[model] = {
            param1_name: linspace1,
            param2_name: linspace2
        }
    
    return linspaces

def evaluate_model_on_data(model, X_train_scaled, y_train_scaled, X_test_scaled, y_test, scaler_y):
    model.fit(X_train_scaled, y_train_scaled)
    y_pred = get_y_original_scale(model, X_test_scaled, scaler_y)
    r2 = r2_score(y_test, y_pred)
    return r2

def shuffle_and_evaluate(model_name: str, file_path: str, X, y, num_shuffles=10, scoring='r2'):
    best_model_info = pt.find_best_models_from_file(file_path)
    if not best_model_info:
        print("No valid model information found.")
        return {}
    
    best_model = best_model_info['model']
    best_params = best_model_info['params']
    best_r2 = best_model_info['r2']
    
    model = clone(best_model).set_params(**best_params)
    
    # Evaluate on original data
    original_score = best_r2
    print(f"Original Score ({scoring}): {original_score}")

    X_train_scaled, X_test_scaled, y_train_scaled, y_test, scaler_X, scaler_y = split_and_standardize(X, y)

    shuffled_scores = []
    for _ in range(num_shuffles):
        X_shuffled, y_shuffled = shuffle(X_train_scaled, y_train_scaled, random_state=42)
        shuffled_score = evaluate_model_on_data(model, X_shuffled, y_shuffled, X_test_scaled, y_test, scaler_y)
        shuffled_scores.append(shuffled_score)

    shuffled_scores = np.array(shuffled_scores)
    mean_shuffled_score = np.mean(shuffled_scores)
    std_shuffled_score = np.std(shuffled_scores)
    conf_interval = stats.t.interval(0.95, len(shuffled_scores)-1, loc=mean_shuffled_score, scale=stats.sem(shuffled_scores))
    
    print(f"Shuffled Scores ({num_shuffles} shuffles): {shuffled_scores}")
    print(f"Mean Shuffled Score: {mean_shuffled_score}")
    print(f"Best Shuffled Score: {shuffled_scores.max()}")
    print(f"Standard Deviation of Shuffled Scores: {std_shuffled_score}")
    print(f"95% Confidence Interval of Shuffled Scores: {conf_interval}")
    print(f"Average difference: {original_score - mean_shuffled_score}")

    results = {
        "original_score": original_score,
        "shuffled_scores": shuffled_scores.tolist(),  # Convert to list for JSON serialization
        "mean_shuffled_score": mean_shuffled_score,
        "best_shuffled_score": shuffled_scores.max(),
        "std_shuffled_score": std_shuffled_score,
        "conf_interval": conf_interval,
        "average_difference": mean_shuffled_score - original_score
    }

    if global_config.get('save_shuffle_data', False):
        save_folder = 'outputs/shuffle'
        os.makedirs(save_folder, exist_ok=True)
        save_file_path = os.path.join(save_folder, f'shuffle_results_{model_name.replace(" ", "_")}.json')
        with open(save_file_path, 'w') as f:
            json.dump(results, f, indent=4)
        print(f"Shuffling results saved to {save_file_path}")

    return results

def create_and_plot_models(X, y):
    models = {
        'Support Vector Regression': SVR,
        'Random Forest': RandomForestRegressor,
        'Feed-Forward Neural Network': MLPRegressor,
        'Gaussian Process': GaussianProcessRegressor,
        'Kernel Ridge Regression': KernelRidge,
        'K-Nearest Neighbor': KNeighborsRegressor,
        'Elastic Net Regression': ElasticNet,
        'Gradient Boosting Regression': GradientBoostingRegressor,
        'AdaBoost Regression': AdaBoostRegressor,
        'Decision Tree Regression': DecisionTreeRegressor
    }

    # Standardize the features
    scaler_X = StandardScaler()
    X_scaled = scaler_X.fit_transform(X)

    # Apply PCA to reduce the feature space to two components
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)

    for model_name, config in model_plot_config.items():
        if not config.get('plot', False):
            continue

        model_class = models[model_name]
        base_params = config['params']
        extended_params = model_plot_config_extended[model_name]['params']
        
        for param_name, param_values in extended_params.items():
            for param_value in param_values:
                model_params = base_params.copy()
                model_params[param_name] = param_value
                model = model_class(**model_params)
                
                # Fit the model on the reduced feature space
                model.fit(X_pca, y)

                # Generate a grid of data points using linspace
                x_min, x_max = X_pca[:, 0].min() - 1, X_pca[:, 0].max() + 1
                y_min, y_max = X_pca[:, 1].min() - 1, X_pca[:, 1].max() + 1
                xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100), np.linspace(y_min, y_max, 100))
                grid_points = np.c_[xx.ravel(), yy.ravel()]

                # Predict the values on the grid data points
                zz = model.predict(grid_points).reshape(xx.shape)

                # Create a 3D plot
                fig = plt.figure(figsize=(10, 7))
                ax = fig.add_subplot(111, projection='3d')

                # Plot the prediction surface with transparency
                ax.plot_surface(xx, yy, zz, cmap='viridis', alpha=0.9)

                # Plot settings
                ax.set_xlabel('Principal Component 1')
                ax.set_ylabel('Principal Component 2')
                ax.set_zlabel('Predicted BC')
                ax.set_title(f'3D Prediction Surface - {model_name} ({param_name}={param_value})')

                # Save the plot
                if global_config.get('allow_save', False):
                    save_folder = model_plot_config_extended[model_name]['save_folder']
                    param_folder = os.path.join(save_folder, param_name)
                    save_filename_template = model_plot_config_extended[model_name]['save_filename']
                    save_filename = save_filename_template.format(**model_params)
                    filepath = os.path.join(param_folder, save_filename)
                    os.makedirs(param_folder, exist_ok=True)
                    plt.savefig(filepath, bbox_inches='tight')
                    print(f'Saved plot to {filepath}')


                # Show plot
                plt.show()
                plt.close()

def plot_model(model, X, y, plot_type, params, scaler_y):
    if plot_type == 'prediction_curve':
        plot_prediction_curve(model, X, y, scaler_y, params)
    elif plot_type == 'feature_importance':
        plot_feature_importance(model)
    elif plot_type == 'training_curve':
        plot_training_curve(model)
    elif plot_type == 'tree_structure':
        plot_tree_structure(model)
    else:
        print(f"Unknown plot type: {plot_type}")

def plot_prediction_curve(model, X, y_true, scaler_y, params):
    # Predict and handle inverse scaling if scaler_y is provided
    y_pred_scaled = model.predict(X)
    if scaler_y:
        y_pred = scaler_y.inverse_transform(y_pred_scaled.reshape(-1, 1)).ravel()
    else:
        y_pred = y_pred_scaled

    # Calculate the R^2 score for better evaluation
    r2_score = model.score(y_pred, y_true)
    
    plt.figure(figsize=(10, 6))
    
    # Scatter plot of true vs predicted values
    sns.scatterplot(x=y_true, y=y_pred, alpha=0.9, edgecolor=None)
    
    # Regression line
    sns.regplot(x=y_true, y=y_pred, scatter=False, color='red', line_kws={'lw': 2})
    
    # Line for ideal predictions
    min_val = min(y_true.min(), y_pred.min())
    max_val = max(y_true.max(), y_pred.max())
    plt.plot([min_val, max_val], [min_val, max_val], 'k--', lw=2, label='Ideal')

    # Add grid
    plt.grid(True, linestyle='--', alpha=0.9)

    # Labels and title
    plt.xlabel('True Value')
    plt.ylabel('Predicted Value')
    plt.title(f"Prediction Curve with {params}\nR^2 Score: {r2_score:.2f}")
    plt.legend()
    
    # Show plot
    plt.tight_layout()
    plt.show()

def plot_feature_importance(model):
    try:
        importances = model.feature_importances_
        indices = np.argsort(importances)[::-1]
        plt.title("Feature importances")
        plt.bar(range(len(importances)), importances[indices], align="center")
        plt.xticks(range(len(importances)), indices)
        plt.xlim([-1, len(importances)])
        plt.show()
    except AttributeError:
        print("The model does not have feature importances.")

def plot_training_curve(model):
    try:
        plt.plot(model.loss_curve_)
        plt.title("Training Curve")
        plt.xlabel("Iterations")
        plt.ylabel("Loss")
        plt.show()
    except AttributeError:
        print("The model does not have a loss curve.")

def plot_tree_structure(model):
    plt.figure(figsize=(20,10))
    plot_tree(model, filled=True, feature_names=None, class_names=None, rounded=True, proportion=False, precision=2)
    plt.title("Tree Structure")
    plt.show()

def is_categorical(data):
    """Determine if the data is categorical."""
    # Check if the first element is a string
    if isinstance(data[0], str):
        return True
    # Check if all elements in the list are strings
    if all(isinstance(item, str) for item in data):
        return True
    # Check if the number of unique elements is less than a threshold
    if len(np.unique(data)) < 10:
        return True
    return False

def flatten_values(values):
    """Flatten nested lists and ensure all values are numeric."""
    flat_values = []
    for value in values:
        if isinstance(value, list) or isinstance(value, tuple):
            flat_values.extend(value)
        else:
            flat_values.append(value)
    return flat_values

def plot_r2_scores(model_name: str, json_file: str):
    """
    Plots the R^2 scores against two parameters for a given model based on configuration.
    
    Parameters:
    - model_name: The name of the model to filter results.
    - json_file: The path to the JSON file containing results.
    """
    
    # Check if the file exists
    if not os.path.isfile(json_file):
        print(f"File {json_file} does not exist. Skipping plotting for {model_name}.")
        return
    
    # Load JSON data
    with open(json_file, 'r') as file:
        data = json.load(file)
    
    # Filter results for the specified model
    model_data = [entry for entry in data if entry['model'] == model_name]
    
    # Apply plot filters from plot configuration
    plot_filter = plot_config[model_name].get('plot_filter', {})
    for key, value in plot_filter.items():
        model_data = [entry for entry in model_data if entry['params'].get(key) == value]
    
    # Get plot parameters from plot configuration
    plot_params = plot_config[model_name].get('plot_params', (None, None))
    x_param, y_param = plot_params
    
    # Extract relevant parameters and R^2 scores
    x_values = []
    y_values = []
    r2_scores = []

    for entry in model_data:
        params = entry['params']
        if x_param in params and y_param in params:
            x_value = params[x_param]
            y_value = params[y_param]

            # Handle nested lists
            if isinstance(x_value, list) or isinstance(y_value, list):
                if isinstance(x_value, list) and not isinstance(y_value, list):
                    y_value = [y_value] * len(x_value)
                elif not isinstance(x_value, list) and isinstance(y_value, list):
                    x_value = [x_value] * len(y_value)
                elif isinstance(x_value, list) and isinstance(y_value, list):
                    if len(x_value) != len(y_value):
                        print(f"Skipping entry due to mismatched nested parameter lengths: {entry}")
                        continue

                for xv, yv in zip(x_value, y_value):
                    x_values.append(xv)
                    y_values.append(yv)
                    r2_scores.append(entry['r2'])
            else:
                x_values.append(x_value)
                y_values.append(y_value)
                r2_scores.append(entry['r2'])
        else:
            print(f"Skipping entry due to missing parameters: {entry}")

    # Flatten values and ensure they are numeric
    x_values = flatten_values(x_values)
    y_values = flatten_values(y_values)

    # Ensure lengths of x_values, y_values, and r2_scores match
    if len(x_values) != len(y_values) or len(x_values) != len(r2_scores):
        print(f"Lengths of x_values, y_values, and r2_scores do not match: {len(x_values)}, {len(y_values)}, {len(r2_scores)}")
        return
    
    # Check if data is categorical
    if is_categorical(x_values) or is_categorical(y_values):
        plot_categorical_data(x_values, y_values, r2_scores, x_param, y_param, model_name)
    else:
        plot_numerical_data(x_values, y_values, r2_scores, x_param, y_param, model_name)

def plot_numerical_data(x_values, y_values, r2_scores, x_param, y_param, model_name):
    # Convert to numpy arrays for easier handling
    x_values = np.array(x_values)
    y_values = np.array(y_values)
    r2_scores = np.array(r2_scores)

    # Check for sufficient variation in the data
    if len(np.unique(x_values)) < 2 or len(np.unique(y_values)) < 2:
        print(f"Not enough variation in the data for {model_name} to create a surface plot.")
        return

    # Create a grid for surface plot
    xi = np.linspace(min(x_values), max(x_values), 100)
    yi = np.linspace(min(y_values), max(y_values), 100)
    xi, yi = np.meshgrid(xi, yi)

    try:
        # Interpolate the r2_scores for the grid
        zi = griddata((x_values, y_values), r2_scores, (xi, yi), method='cubic')
    except Exception as e:
        print(f"Error during interpolation for {model_name}: {e}")
        return

    # Plotting
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Create surface plot
    surf = ax.plot_surface(xi, yi, zi, cmap=cm.viridis, edgecolor='none')

    # Add color bar which maps values to colors
    fig.colorbar(surf, ax=ax, label='R^2 Score')

    # Set labels and title
    ax.set_xlabel(x_param)
    ax.set_ylabel(y_param)
    ax.set_zlabel('R^2 Score')
    plt.title(f'R^2 Scores for {model_name}', fontsize=15)

    # Customize the view angle
    ax.view_init(elev=20., azim=30)

    # Show plot
    plt.show()

def plot_categorical_data(x_values, y_values, r2_scores, x_param, y_param, model_name):
    import seaborn as sns
    
    # Convert all values to strings to handle mixed types
    x_values = [str(x) for x in x_values]
    y_values = [str(y) for y in y_values]
    
    # Create a DataFrame for easier plotting
    data = pd.DataFrame({
        x_param: x_values,
        y_param: y_values,
        'R^2 Score': r2_scores
    })
    
    # Aggregate duplicates by averaging the R^2 scores
    data_agg = data.groupby([y_param, x_param]).mean().reset_index()
    
    # Plotting
    plt.figure(figsize=(12, 8))
    sns.barplot(data=data_agg, x=x_param, y='R^2 Score', hue=y_param)
    
    # Set axis labels and title
    plt.xlabel(x_param)
    plt.ylabel('R^2 Score')
    plt.title(f'R^2 Scores for {model_name}', fontsize=15)
    plt.legend(title=y_param)
    plt.show()

def data_extraction(data_path):
    # Load data
    data = pd.read_csv(data_path, delimiter=';')
    # Drop the 'date' column as it is not needed for prediction
    data = data.drop(columns=['date'])
    # Separate the features and the target variable
    X = data.drop(columns=['BC'])
    y = data['BC']
    return data, X, y

def split_and_standardize(X, y):
    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Standardize the features
    scaler_X = StandardScaler()
    X_train_scaled = scaler_X.fit_transform(X_train)
    X_test_scaled = scaler_X.transform(X_test)
    # Standardize the target variable
    scaler_y = StandardScaler()
    y_train_scaled = scaler_y.fit_transform(y_train.values.reshape(-1, 1)).ravel()
    
    return X_train_scaled, X_test_scaled, y_train_scaled, y_test, scaler_X, scaler_y

def get_y_original_scale(model, X_test, scaler_y):
    y_pred = model.predict(X_test)
    return scaler_y.inverse_transform(y_pred.reshape(-1, 1)).ravel()

