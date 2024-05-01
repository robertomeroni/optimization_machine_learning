import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error



def evaluate_predictions(y_true, y_pred):
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    return rmse, r2, mae



def get_neighborhood_data(node_index, adjacency_matrix, data_matrix, print_neighbors=True):
    adjacency_matrix[node_index, node_index] = 0
    neighbors = np.where(adjacency_matrix[node_index, :] != 0)[0]
    num_neighbors = len(neighbors)
    if num_neighbors == 0:
        # All other nodes are considered as neighbors, except itself
        neighbors = np.setdiff1d(np.arange(adjacency_matrix.shape[0]), [node_index])
        num_neighbors = len(neighbors)  
    if print_neighbors:
        print(f"Node {data_matrix.columns[node_index]} has {num_neighbors} neighbors.")
    neighbor_data = data_matrix.iloc[:, neighbors]
    weights = adjacency_matrix[node_index, neighbors]
    if np.sum(weights) <= abs(1e-3):
        weights = np.ones(num_neighbors) / num_neighbors
    else:
        weights = weights / np.sum(weights)
    if np.sum(weights) <= 0.99 or np.sum(weights) >= 1.01:
        print(f"Warning: Weights for node {data_matrix.columns[node_index]} do not sum to 1.")
        print(f"Sum of weights: {np.sum(weights)}")
    return neighbor_data, data_matrix.iloc[:, node_index], weights

def signal_reconstruction(data, adjacency_matrix, print_neighbors=True, print_sensors=False):
    results = {}
    total_relevant_weights = []
    num_nodes = data.shape[1]
    num_edges = (np.count_nonzero(adjacency_matrix) - np.count_nonzero(np.diagonal(adjacency_matrix))) / 2
    
    for node_index in range(num_nodes):
        X, y, weights = get_neighborhood_data(node_index, adjacency_matrix, data, print_neighbors=print_neighbors)
        
        relevant_weights = [weights[i] for i in range(len(weights)) if adjacency_matrix[node_index, i] != 0 and i != node_index]
        total_relevant_weights.extend(relevant_weights)

        y_pred = X @ weights
        rmse, r2, mae = evaluate_predictions(y, y_pred)
        # Store results
        results[data.columns[node_index]] = {'RMSE': rmse, 'R^2': r2, 'MAE': mae}

    # Display results
    if print_sensors:
        for sensor, metrics in results.items():
            print(f"Sensor: {sensor}, RMSE: {metrics['RMSE']:.3f}, R^2: {metrics['R^2']:.3f}")

    # Compute average RMSE and R^2
    if results:
        average_rmse = sum(d['RMSE'] for d in results.values()) / len(results)
        average_r2 = sum(d['R^2'] for d in results.values()) / len(results)
    else:
        average_rmse = 0
        average_r2 = 0
    
    # Calculate the average of the relevant weights
    if total_relevant_weights:
        average_weight = sum(total_relevant_weights) / len(total_relevant_weights)
    else:
        average_weight = 0

    return results, average_rmse, average_r2, average_weight, num_edges



