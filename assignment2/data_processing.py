import json
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import networkx as nx
from scipy.interpolate import griddata

def data_extraction(data_path):
    # Load data
    data = pd.read_csv(data_path, delimiter=';')
    # Drop the 'date' column
    if 'date' in data.columns:
        data = data.drop(columns=['date'])
    # Specify the column order
    column_order = ['gracia', 'hebron', 'pr', 'prat', 'ciutadella', 'eixample', 'badalona', 'montcada']
    # Reorder the columns
    data = data[column_order]
    return data



def data_standardization(data):
    # Initialize the StandardScaler
    scaler = StandardScaler()
    # Scale the features
    data_scaled = scaler.fit_transform(data)
    return pd.DataFrame(data_scaled, columns=data.columns)



def matrix_flattening(matrix):
    # Flatten the matrix and filter out zero (self-distances) and redundant elements (above the diagonal)
    flattened_matrix = [matrix[i][j] for i in range(len(matrix)) for j in range(i+1, len(matrix))]
    return flattened_matrix


def haversine(lat1, lon1, lat2, lon2):
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
    c = 2 * np.arcsin(np.sqrt(a))
    R = 6371  # Earth radius in kilometers
    return R * c


def convert_distances(data_path):
    # Load data
    df = pd.read_csv(data_path, delimiter=';')
    lat_column = 'Lat' 
    lon_column = 'Lon' 

    # Calculate the distances
    num_nodes = len(df)
    distances = np.zeros((num_nodes, num_nodes))  # Initialize a matrix to store distances

    for i in range(num_nodes):
        for j in range(num_nodes):
            if i != j:
                distances[i, j] = haversine(df[lat_column][i], df[lon_column][i], df[lat_column][j], df[lon_column][j])

    return distances


def store_and_save_results_to_json(all_model_results):
    # Convert results to JSON and save to file
    with open('model_results.json', 'w') as f:
        json.dump(all_model_results, f, indent=4, default=str)  # Use default=str to handle any non-serializable data like NumPy arrays

def load_results_from_json(file_path):
    # Open the JSON file and load data
    with open(file_path, 'r') as f:
        all_model_results = json.load(f)
    return all_model_results

def plot_graph(file_path, weight_matrix, vmin=-1.0, vmax=1.0):
    # Load node locations from a CSV file
    df = pd.read_csv(file_path, delimiter=';')
    positions = {row['Name']: (row['Lon'], row['Lat']) for index, row in df.iterrows()}
    
    # Create a graph
    G = nx.Graph()
    
    # Add nodes with positions
    for node, pos in positions.items():
        G.add_node(node, pos=pos)
    
    # Add edges based on the weight matrix
    nodes = list(positions.keys())
    for i in range(len(nodes)):
        for j in range(i + 1, len(nodes)):
            weight = weight_matrix[i][j]
            if weight != 0:  # Only add edges with non-zero weights
                G.add_edge(nodes[i], nodes[j], weight=weight)
    
    # Prepare the plot
    fig, ax = plt.subplots(figsize=(10, 8))
    pos = {node: positions[node] for node in G.nodes()}
    
    # Extract edge weights
    edge_weights = np.array([G[u][v]['weight'] for u, v in G.edges()])

    # Create a color map normalized to the edge weights
    norm = plt.Normalize(vmin=vmin, vmax=vmax)
    cmap = plt.cm.coolwarm

    # Draw the network
    nx.draw_networkx(G, pos, ax=ax, node_color='skyblue', node_size=500, 
                     edge_color=edge_weights, width=2, edge_cmap=cmap, edge_vmin=vmin, edge_vmax=vmax, with_labels=True)

    # Create the colorbar
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    plt.colorbar(sm, ax=ax, orientation='vertical', label='Edge weight')
    
    plt.title("Graph of Nodes Based on Weighted Connections")
    plt.show()


def plot_graph2(file_path, weight_matrix, vmin=-1.0, vmax=1.0):
    # Load node locations from a CSV file
    df = pd.read_csv(file_path, delimiter=';')
    positions = {row['Name']: (row['Lon'], row['Lat']) for index, row in df.iterrows()}
    print(weight_matrix)
    # Create a graph
    G = nx.Graph()
    
    # Add nodes with positions
    for node, pos in positions.items():
        G.add_node(node, pos=pos)
    
    # Add edges based on the weight matrix
    nodes = list(positions.keys())
    for i in range(len(nodes)):
        for j in range(i + 1, len(nodes)):
            if weight_matrix[i][j] != 0:  # Use weight matrix to decide on the presence of an edge
                G.add_edge(nodes[i], nodes[j], weight=weight_matrix[i][j])
    
    # Prepare the plot
    fig, ax = plt.subplots(figsize=(10, 8))
    pos = {node: positions[node] for node in G.nodes()}
    weights = np.array([G[u][v]['weight'] for u, v in G.edges()])
    
    # Draw the graph using weights for edge visualization
    nx.draw_networkx(G, pos, ax=ax, node_color='skyblue', node_size=500, 
                     edge_color=weights, width=2, edge_cmap=plt.cm.Blues, with_labels=True)
    
    # Create the colorbar
    sm = plt.cm.ScalarMappable(cmap=plt.cm.Blues, norm=plt.Normalize(vmin=vmin, vmax=vmax))
    sm.set_array([])
    plt.colorbar(sm, ax=ax, orientation='vertical', label='Edge weight')
    
    plt.title("Graph of Nodes Based on Weighted Connections")
    plt.show()

def plot3d(all_model_results, metric='r2'):
    # Validate input
    valid_metrics = {'r2', 'num_edges', 'average_weight'}
    if metric not in valid_metrics:
        raise ValueError(f"Invalid metric. Choose from: {valid_metrics}")
    
    # Set up the figure for plotting
    model_types = set(key.split('_')[0] for key in all_model_results.keys())
    fig = plt.figure(figsize=(10, 8))
    
    for i, model in enumerate(sorted(model_types), start=1):
        ax = fig.add_subplot(1, len(model_types), i, projection='3d')
        ax.set_title(model)
        
        points = []
        values = []
        param_names = []
        
        # Collect data based on chosen metric
        for key, info in all_model_results.items():
            if key.startswith(model):
                params = info['parameters']
                param_keys = list(params.keys())
                if len(param_keys) >= 2:
                    point = (params[param_keys[0]], params[param_keys[1]])
                    points.append(point)
                    if metric == 'r2':
                        values.append(info['average_r2'])
                    elif metric == 'num_edges':
                        values.append(info.get('num_edges', 0))  # Default to 0 if not found
                    elif metric == 'average_weight':
                        values.append(info.get('average_weight', 0))  # Default to 0 if not found
                    if not param_names:
                        param_names = param_keys

        ax.set_xlabel(param_names[0] if len(param_names) > 0 else 'Parameter 1')
        ax.set_ylabel(param_names[1] if len(param_names) > 1 else 'Parameter 2')
        ax.set_zlabel(metric.replace('_', ' ').title())
        
        points = np.array(points)
        values = np.array(values)

        # Interpolate data for smoother surface plotting
        grid_x, grid_y = np.mgrid[min(points[:, 0]):max(points[:, 0]):100j, min(points[:, 1]):max(points[:, 1]):100j]
        grid_z = griddata(points, values, (grid_x, grid_y), method='cubic')
        
        surface = ax.plot_surface(grid_x, grid_y, grid_z, cmap='viridis', edgecolor='none')
        fig.colorbar(surface, ax=ax, shrink=0.5, aspect=5)

    plt.tight_layout()
    plt.show()