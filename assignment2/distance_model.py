import numpy as np
import data_processing as dp
from data_paths import location_data_path 

def weight_function(dist, theta, T):
    """
    Calculate the weight between two points using a Gaussian decay function based on their Euclidean distance.
    
    Parameters:
    - dist: Euclidean distance between the two points.
    - theta: Decay parameter of the Gaussian function.
    - T: Distance threshold above which the weight is zero.
    
    Returns:
    - Weight between the two points as a float.
    """
    
    # Apply the weight function based on the distance
    if dist < T:
        weight = np.exp(-dist**2 / (2 * theta**2))
    else:
        weight = 0
    
    return weight

def distance_model(distances, theta, T):
    num_points = len(distances)
    weights_matrix = np.zeros((num_points, num_points))
    
    for i in range(num_points):
        for j in range(i + 1, num_points):  
            dist = distances[i, j]
            weight = weight_function(dist, theta, T)
            weights_matrix[i, j] = weight
            weights_matrix[j, i] = weight  

    return weights_matrix


