import numpy as np
import pandas as pd
import data_processing as dp
import distance_model as dm
import glasso_model as gm
import laplacian_model as lm
import signal_reconstruction as sr
from data_paths import measurements_data_path, location_data_path



# Distance-based model
def distance_based_model(data, distances, thetas, Ts, graph=False, print_matrix=True):
    model_results = []
    for theta in thetas:
        for T in Ts:
            weights_matrix = dm.distance_model(distances, theta, T)
            weights_df = pd.DataFrame(weights_matrix, columns=data.columns, index=data.columns)
            if print_matrix:
                print("Theta: ", theta)
                print("T: ", T)
                print("Weights matrix:")
                print(weights_df)
                print()
            if graph:
                dp.plot_graph2(location_data_path, weights_matrix, vmin=0.0)
            model_results.append(("Distance-based model", weights_matrix))
    return model_results


# GLASSO model
def glasso_based_model(data, lambdas=[0.001], Ts=[0], sklearn=True, cvxpy=False, compare=False, graph=False, print_matrix=False, info=False):
    model_results = []
    ## GLASSO with sklearn
    if sklearn:
        for lmbda in lambdas:
            for T in Ts:
                precision_matrix = gm.glasso_model(data, lmbda, T)
                # Convert the thresholded precision matrix to a DataFrame for better readability
                precision_df = pd.DataFrame(precision_matrix, columns=data.columns, index=data.columns)
                if print_matrix:
                    print("GLASSO Precision matrix:")
                    print(precision_df)
                    print()
                if graph:
                    dp.plot_graph(location_data_path, precision_matrix, vmin=-1.0)
                if info:
                    print("Edges in precision matrix: ", np.count_nonzero(np.triu(precision_matrix, k=1)))
                model_results.append(("sklearn GLASSO model", precision_matrix))

    ## GLASSO with CVXPY
    if cvxpy:
        for lmbda in lambdas:
            for T in Ts:
                cvxpy_precision_matrix = gm.glasso_cvxpy(data, lmbda, T)
                if print_matrix:
                    print("CVXPY GLASSO Precision matrix:")
                    print(cvxpy_precision_matrix)
                    print()
                model_results.append(("CVXPY GLASSO model", cvxpy_precision_matrix))

    ## Compare the two precision matrices
    if compare:
        for T in Ts:
            print()
            print("Threshold: ", T)
            for lmbda in lambdas:
                precision_matrix = gm.glasso_model(data, lmbda, T)
                cvxpy_precision_matrix = gm.glasso_cvxpy(data, lmbda, T)
                if print_matrix:
                    print("GLASSO Precision matrix:")
                    print(precision_matrix)
                if print_matrix:
                    print("CVXPY GLASSO Precision matrix:")
                    print(cvxpy_precision_matrix)
                
                diff = np.abs(precision_matrix - cvxpy_precision_matrix)
                frobenius_norm = np.linalg.norm(diff, 'fro')
                print()
                print("lmbda: ", lmbda)
                print("Edges in precision matrix: ", np.count_nonzero(np.triu(precision_matrix, k=1)))
                print("Edges in CVXPY precision matrix: ",np.count_nonzero(np.triu(cvxpy_precision_matrix, k=1)))
                print("Frobenius Norm of Difference between the two precision matrices: ", frobenius_norm)
    return model_results

def laplacian_based_model(data, alphas=[0.0001], betas=[0.1], tol=1e-9, print_matrix=False, graph=False):
    model_results = []
    data_np = data.to_numpy()
    data_np = data_np.T
    for alpha in alphas:
        print("Alpha: ", alpha)
        for beta in betas:
            print("Beta: ", beta)
            adjacency_matrix = lm.laplacian_model(data_np, alpha=alpha, beta=beta, max_iter=1000, tol=tol)
            if print_matrix:
                df = pd.DataFrame(adjacency_matrix, columns=data.columns, index=data.columns)
                print("Adjacency Matrix from Laplacian:")
                print(df.to_string())
                print()
            if graph:
                dp.plot_graph2(location_data_path, adjacency_matrix, vmin=-1.0)
            model_results.append(("Laplacian model", adjacency_matrix))
    return model_results

def test_results(data, distances, thetas, Ts, lambdas, Thresholds, alphas, betas, distance=False, glasso=False, laplacian=False, graph=False, print_matrix=False):
    models = []
    if distance:
        for theta in thetas:
            for T in Ts:
                model_outputs = distance_based_model(data, distances, [theta], [T], graph=graph, print_matrix=print_matrix)
                for description, model in model_outputs:
                    models.append((description, model, {'theta': theta, 'T': T}))
    if glasso:
        for lmbda in lambdas:
            for Threshold in Thresholds:
                model_outputs = glasso_based_model(data, lambdas=[lmbda], Ts=[Threshold], sklearn=True, cvxpy=False, compare=False, graph=graph, print_matrix=print_matrix, info=False)
                for description, model in model_outputs:
                    models.append((description, model, {'lambda': lmbda, 'Threshold': Threshold}))
    if laplacian:
        for alpha in alphas:
            for beta in betas:
                model_outputs = laplacian_based_model(data, alphas=[alpha], betas=[beta], tol=1e-4, print_matrix=print_matrix, graph=graph)
                for description, model in model_outputs:
                    models.append((description, model, {'alpha': alpha, 'beta': beta}))
    return models

def find_best_parameters(all_model_results):
    # Dictionary to store the best performance for each model
    best_performance = {}
    # Dictionary to track average R^2 for each node in each model
    node_r2_averages = {}
    # Dictionary to track overall average R^2 for each model
    model_r2_totals = {}

    # Iterate through all the model results stored in the dictionary
    for key, value in all_model_results.items():
        # Extract model description from the key (assuming it's the first part before '_')
        model_description = key.split("_")[0]

        # Initialize the model type in the dictionary if not already present
        if model_description not in best_performance:
            best_performance[model_description] = {
                'best_r2': {'value': -float('inf'), 'params': None, 'num_edges': 0, 'rmse': float('inf')},
                'best_rmse': {'value': float('inf'), 'params': None, 'num_edges': 0, 'r2': -float('inf')},
                'best_node_r2': {'node': None, 'average_r2': -float('inf')}
            }
        if model_description not in node_r2_averages:
            node_r2_averages[model_description] = {}
        if model_description not in model_r2_totals:
            model_r2_totals[model_description] = {'total_r2': 0, 'count': 0}

        # Iterate through each node's results
        for node, node_results in value['results'].items():
            node_r2 = node_results['R^2']
            # Update total R^2 and count for the model
            model_r2_totals[model_description]['total_r2'] += node_r2
            model_r2_totals[model_description]['count'] += 1

            if node not in node_r2_averages[model_description]:
                node_r2_averages[model_description][node] = {'total_r2': 0, 'count': 0}
            node_r2_averages[model_description][node]['total_r2'] += node_r2
            node_r2_averages[model_description][node]['count'] += 1
            
            # Update the best R^2 and RMSE as before
            if node_r2 > best_performance[model_description]['best_r2']['value']:
                best_performance[model_description]['best_r2'] = {
                    'value': node_r2,
                    'params': value['parameters'],
                    'num_edges': value['num_edges'],
                    'rmse': node_results['RMSE']
                }
            current_rmse = node_results['RMSE']
            if current_rmse < best_performance[model_description]['best_rmse']['value']:
                best_performance[model_description]['best_rmse'] = {
                    'value': current_rmse,
                    'params': value['parameters'],
                    'num_edges': value['num_edges'],
                    'r2': node_r2
                }

    # Calculate average R^2 for each node and update best node R^2
    for model, nodes in node_r2_averages.items():
        for node, data in nodes.items():
            average_r2 = data['total_r2'] / data['count']
            if average_r2 > best_performance[model]['best_node_r2']['average_r2']:
                best_performance[model]['best_node_r2'] = {
                    'node': node,
                    'average_r2': average_r2
                }

    # Print the results
    for model, perf in best_performance.items():
        overall_average_r2 = model_r2_totals[model]['total_r2'] / model_r2_totals[model]['count']
        print(f"Model: {model}")
        print(f"  Best R^2: {perf['best_r2']['value']:.3f}")
        print(f"    Parameters: {perf['best_r2']['params']}")
        print(f"    Number of Edges: {perf['best_r2']['num_edges']}")
        print(f"    Associated RMSE: {perf['best_r2']['rmse']:.3f}")
        print(f"  Best node by average R^2: {perf['best_node_r2']['node']} with R^2: {perf['best_node_r2']['average_r2']:.3f}")
        print(f"  Overall average R^2 across all parameters: {overall_average_r2:.3f}")
        print("-" * 40)



def store_model_results(data, distances, thetas, Ts, lambdas, Thresholds, alphas, betas, distance=True, glasso=0, laplacian=0, graph=False, print_matrix=False):
    # Generate the models with various parameters
    models = test_results(data, distances, thetas, Ts, lambdas, Thresholds, alphas, betas, distance=distance, glasso=glasso, laplacian=laplacian, graph=graph, print_matrix=print_matrix)
    
    # Dictionary to store all results
    all_model_results = {}
    
    # Iterate through each model and its output
    for model_description, model_output, parameters in models:
        # Create a unique key by combining the model description and parameters
        # Format parameters into a string to be part of the key
        params_key = "_".join(f"{key}={value}" for key, value in parameters.items())
        unique_key = f"{model_description}_{params_key}"
        
        # Perform signal reconstruction and store the results
        results, average_rmse, average_r2, average_weight, num_edges = sr.signal_reconstruction(data, model_output, print_neighbors=False, print_sensors=False)
        
        # Save results and parameters under the unique key
        all_model_results[unique_key] = {
            'results': results,
            'num_edges': num_edges,  
            'average_weight': average_weight,  
            'average_rmse': average_rmse,
            'average_r2': average_r2,
            'parameters': parameters
        }
        dp.store_and_save_results_to_json(all_model_results)
    return all_model_results

def print_model_results(all_model_results, detailed_results=False):
    # Iterate through all entries in the results dictionary
    for unique_key in all_model_results:
        # Retrieve the stored information for the current model
        model_info = all_model_results[unique_key]
        # Print the unique key (model and parameters)
        print(f"Model and Parameters: {unique_key}")
        
        # Print the results metrics 
        print(f"Number of edges: {model_info['num_edges']}")
        print(f"Average Weight: {model_info['average_weight']:.3f}")
        print(f"Average RMSE: {model_info['average_rmse']:.3f}")
        print(f"Average R^2: {model_info['average_r2']:.3f}")
        
        if detailed_results:
            print("Detailed Results:")
            for sensor, metrics in model_info['results'].items():
                rmse = metrics['RMSE']
                r2 = metrics['R^2']
                mae = metrics.get('MAE', 0)  # Defaulting MAE to 0 if not available
                print(f"  Sensor: {sensor}, RMSE: {rmse:.3f}, R^2: {r2:.3f}, MAE: {mae:.3f}")
        print("-" * 60)

# Data extraction and standardization.
data = dp.data_extraction(measurements_data_path)
data = dp.data_standardization(data)
distances = dp.convert_distances(location_data_path)
distances_flattened = dp.matrix_flattening(distances)
models = []
all_model_results = {}

# Set values.
num_values = 10
## For distance-based model
thetas = [p for p in np.logspace(0.01, 4.17, num_values)]
Ts = [np.max(distances_flattened)*p for p in np.linspace(0.2, 1, num_values)]
## For GLASSO model
lambdas = [p for p in np.linspace(0.0001, 1, num_values)]
Thresholds = [p for p in np.linspace(0.1, 1, num_values)]

## For Laplacian model
alphas = [p for p in np.linspace(0.0001, 0.001, num_values)]
betas = [p for p in np.linspace(0.2, 1, num_values)]

all_model_results = store_model_results(data, distances, thetas, Ts, lambdas, Thresholds, alphas, betas, distance=1, glasso=0, laplacian=0, graph=False, print_matrix=0)
# all_model_results = dp.load_results_from_json('model_results_laplacian.json')
print_model_results(all_model_results, detailed_results=False)
find_best_parameters(all_model_results)
dp.plot3d(all_model_results, metric='num_edges')
dp.plot3d(all_model_results, metric='average_weight')
dp.plot3d(all_model_results, metric='r2')
