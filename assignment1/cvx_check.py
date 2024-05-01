import numpy as np
import numdifftools as nd
import time
from scipy.optimize import minimize
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def optimize_with_timing(initial_guesses, objective_function, constraints, jacobian=None):
    # Empty lists 
    timings_with_jacobian = []
    timings_no_jacobian = []

    # Callback function to print the current x values
    def callback(x):
        x_values.append(x)
        # print(f"Current x: {x}, Objective function value: {objective_function(x)}, jacobian: {nd.Jacobian(objective_function)(x)}")


    # Inside your loop where you call the optimization function
    for x0 in initial_guesses:
        x_values = [x0]
        start_time = time.perf_counter()  # Start the timer
        if jacobian:
            result = minimize(objective_function, x0, method='SLSQP', jac=jacobian, constraints=constraints)
        else:
            result = minimize(objective_function, x0, method='SLSQP', constraints=constraints, callback=callback)
        end_time = time.perf_counter()  # End the timer
        
        elapsed_time = end_time - start_time  # Calculate elapsed time
        if jacobian:
            timings_with_jacobian.append(elapsed_time)
        else:
            timings_no_jacobian.append(elapsed_time)
        

        
        print(f"Initial guess: {x0}")
        print(f"{result}") 
        print(f"Time: {end_time - start_time}\n")
        if result.success:
            is_constraint_valid(result.x, constraints)

        # plot_solver_path(x_values, objective_function, constraints)

    return timings_with_jacobian if jacobian else timings_no_jacobian

def is_psd(functions, dim=2, tol=1e-8):
    """
    Evaluates the Hessian of each function at the origin and categorizes it.

    Parameters:
    - functions: A list of functions whose Hessians are to be evaluated. Each function must take a single argument (numpy array) and return a scalar.
    - dim: The dimension of the input to each function, defaulting to 2.

    Prints the categorization of each function's Hessian.
    """

    # Set origin based on dimension
    x0 = np.zeros(dim)

    for func in functions:
        # Create a Hessian function object and evaluate it at x0
        H = nd.Hessian(func)(x0)

        # Compute the eigenvalues of the Hessian
        eigenvalues = np.linalg.eigvals(H)

        # Optional: Print the Hessian and the eigenvalues
        print('Hessian of ' + func.__name__ + ':')
        print(H)
        print('Eigenvalues of the Hessian:')
        print(eigenvalues)

        # Determine the categorization based on eigenvalues
        if np.all(eigenvalues > tol):
            color_start = "\033[92m"  # Green
            status = 'Positive definite'
        elif np.all(eigenvalues >= - tol):
            color_start = "\033[92m"  # Green
            status = 'Positive semi-definite'
        elif np.all(eigenvalues < - tol):
            color_start = "\033[91m"  # Red
            status = 'Negative definite'
        elif np.all(eigenvalues <= tol):
            color_start = "\033[91m"  # Red
            status = 'Negative semi-definite'
        else:
            color_start = "\033[0m"  # Neutral
            status = 'Indefinite'
        
        color_end = "\033[0m"
        print(f"{color_start}{func.__name__} Hessian is: {status}{color_end}")
    print()       


# Check correctness of constraints
def is_constraint_valid(x, constraints, tol=1e-6):
    
    # Iterate over the list of constraint dictionaries
    for i, constraint in enumerate(constraints, start=1):
        # Evaluate the constraint function at the solution x
        constraint_value = constraint['fun'](x)
        
        # Check the type of constraint and determine if it is satisfied
        if constraint['type'] == 'ineq':
            # For inequality, it is satisfied if constraint_value <= 0 within the tolerance
            is_satisfied = constraint_value >= -tol
            constraint_type = "Inequality"
        elif constraint['type'] == 'eq':
            # For equality, use np.isclose to check if it's approximately equal to 0
            is_satisfied = np.isclose(constraint_value, 0, atol=tol)
            constraint_type = "Equality"
        
        # If not satisfied, print a warning message
        if not is_satisfied:
            print(f"\033[91mWarning: Constraint {i} ({constraint_type}) is not satisfied at the solution x={x}\033[0m")
            print(f"\033[91m  Constraint value: {constraint_value}\n\033[0m")

        



# This wrapper will allow f1 and f2 to process meshgrid arrays element-wise
def apply_func_elementwise(func, X1, X2):
    # Prepare an empty container for the output
    Z = np.zeros(X1.shape)
    # Apply the function element-wise
    for i in range(X1.shape[0]):
        for j in range(X1.shape[1]):
            Z[i, j] = func(np.array([X1[i, j], X2[i, j]]))
    return Z


# Wrapper function, unchanged
def apply_func_elementwise(func, X1, X2):
    Z = np.zeros(X1.shape)
    for i in range(X1.shape[0]):
        for j in range(X1.shape[1]):
            Z[i, j] = func(np.array([X1[i, j], X2[i, j]]))
    return Z

def plot_feasible_region(functions):
    # Define the grid for visualization
    x1 = np.linspace(-10, 10, 400)
    x2 = np.linspace(-10, 10, 400)
    X1, X2 = np.meshgrid(x1, x2)

    # Initialize the plot
    plt.figure(figsize=(10, 8))

    # Colors for each function
    colors = ['blue', 'red', 'green', 'purple', 'orange', 'brown', 'pink', 'gray', 'olive', 'cyan']
    labels = ['Function 1', 'Function 2', 'Function 3', 'Function 4', 'Function 5', 'Function 6', 'Function 7', 'Function 8', 'Function 9', 'Function 10']


    # Initialize an array to hold the feasibility of each point
    # Start with all zeros (indicating non-feasible)
    feasible_area = np.zeros(X1.shape)

    for i, func in enumerate(functions):
        # Apply the function over the grid to get the constraint area
        Z = apply_func_elementwise(func, X1, X2)
        # Update the feasible area for the current constraint
        # We consider a point feasible if Z <= 0 (<=0 for <= constraints)
        constraint_feasible = Z <= 0
        # For the first function, initialize feasible_area, else find the intersection
        feasible_area = feasible_area + constraint_feasible
        # Contour plot for the function
        plt.contour(X1, X2, Z, levels=[0], colors=colors[i % len(colors)])
        # Filling the region below/above the contour
        plt.contourf(X1, X2, Z, levels=[-np.inf, 0], alpha=0.5, colors=colors[i % len(colors)])
        # Adding to legend
        plt.plot([], [], color=colors[i % len(colors)], alpha=0.5, label=labels[i % len(labels)])
        
    # Highlight the intersection of all constraints (feasible_area equals the number of functions)
    plt.contourf(X1, X2, feasible_area, levels=[len(functions)-0.5, len(functions)], alpha=0.5, colors=['lightgreen'])

    plt.xlim(-10, 10)
    plt.ylim(-10, 10)
    plt.xlabel('$x_1$')
    plt.ylabel('$x_2$')
    plt.title('Feasible Regions for the Given Functions')
    plt.grid(True)
    plt.legend()
    plt.show()


def plot_solver_path(x_values, objective_function, constraints, buffer=1.0, resolution=50):
    """
    Plots the optimization path along with the objective function surface in 3D space, adapting the range based on x_values,
    and includes the visualization of constraints defined in the scipy.optimize format.
    
    Parameters:
    - x_values: A list or array of x values collected during optimization.
    - objective_function: The objective function used in the optimization.
    - constraints: A list of constraints in the format used by scipy.optimize.
    - buffer: Additional space around the min and max of x_values to ensure the surface is well visualized.
    - resolution: The resolution of the grid for plotting the objective function surface and constraints.
    """
    x_values = np.array(x_values)  # Ensure x_values is a NumPy array for easy slicing
    z_values = [objective_function(x) for x in x_values]  # Objective function values for the path
    
    # # Dynamic range calculation based on x_values
    # x1_min, x1_max = x_values[:, 0].min() - buffer, x_values[:, 0].max() + buffer
    # x2_min, x2_max = x_values[:, 1].min() - buffer, x_values[:, 1].max() + buffer
    
    # Static range for the assignment
    x1_min, x1_max = -10, 10
    x2_min, x2_max = -10, 10
    
    # Create a mesh grid for the objective function surface and constraints
    x1 = np.linspace(x1_min, x1_max, resolution)
    x2 = np.linspace(x2_min, x2_max, resolution)
    X1, X2 = np.meshgrid(x1, x2)
    Z = np.array([objective_function([x, y]) for x, y in zip(np.ravel(X1), np.ravel(X2))])
    Z = Z.reshape(X1.shape)
    
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    
    # Plot the objective function surface
    ax.plot_surface(X1, X2, Z, alpha=0.5, cmap='viridis', edgecolor='none')
    
    # Overlay the optimization path
    ax.plot(x_values[:, 0], x_values[:, 1], z_values, marker='o', color='r', label='Optimization Path')
    
    # Process and plot constraints
    for constraint in constraints:
        constraint_function = constraint['fun']
        C = np.array([constraint_function([x, y]) for x, y in zip(np.ravel(X1), np.ravel(X2))])
        C = C.reshape(X1.shape)
        # Assuming 'ineq' type constraints. For 'eq', adjust the level and maybe use a different contour color.
        ax.contour(X1, X2, C, levels=[0], colors=['green'], linestyles='dashed', alpha=0.5)
    
    ax.set_xlabel('$x_1$')
    ax.set_ylabel('$x_2$')
    ax.set_zlabel('Objective Function Value')
    plt.title('Optimization Path with Constraints')
    plt.legend()
    plt.show()
