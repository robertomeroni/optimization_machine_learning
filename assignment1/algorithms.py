import numpy as np
import time
from scipy.optimize import approx_fprime
import numdifftools as nd

def print_results(x0, x, func_value, iterations, time_taken, solver_type):
    """
    Prints the results of the optimization including the time taken, the type of solver used, and the starting point.

    Parameters:
    - x0: The starting point of the optimization.
    - x: The optimized variable.
    - func_value: The value of the function at the optimized variable.
    - iterations: The number of iterations performed.
    - time_taken: The time taken to perform the optimization.
    - solver_type: The type of solver used ('Newton' or 'Gradient Descent').
    """
    print(f"Solver type: {solver_type}")
    print(f"Starting point (x0): {x0}")
    print(f"Optimized variable (x): {x}")
    print(f"Function value at optimal point (f(x)): {func_value}")
    print(f"Number of iterations performed: {iterations}")
    print(f"Time taken for optimization: {time_taken} seconds")
    print()

def newton_method(f, x0, f_prime=None, f_double_prime=None, precision=1e-4, max_iterations=1000, epsilon=1e-8):
    start_time = time.time()
    x = x0
    for i in range(max_iterations):
        if f_prime is None:
            grad = approx_fprime(np.array([x]), f, epsilon)[0]
        else:
            grad = f_prime(x)
        
        if f_double_prime is None:
            H = nd.Hessian(f)([x])[0][0]
        else:
            H = f_double_prime(x)
        
        if H == 0:
            print("Hessian is zero, stop iteration.")
            break
        
        x_new = x - grad / H
        if np.abs(f(x_new) - f(x)) < precision:
            end_time = time.time()
            print_results(x0, x_new, f(x_new), i + 1, end_time - start_time, "Newton")
            return x_new, f(x_new), i + 1
        x = x_new
    
    end_time = time.time()
    print_results(x0, x, f(x), max_iterations, end_time - start_time, "Newton")
    return x, f(x), max_iterations

def gradient_descent(func, x0, grad_func=None, alpha=0.3, beta=0.8, precision=1e-4, max_iterations=1000, epsilon=1e-8):
    start_time = time.time()
    x = np.array([x0])
    for i in range(max_iterations):
        if grad_func is not None:
            grad = grad_func(x)
        else:
            grad = approx_fprime(x, func, epsilon)
        
        t = 1.0
        while func(x - t * grad) > func(x) - alpha * t * np.dot(grad, grad):
            t *= beta
        x_new = x - t * grad
        if np.abs(func(x_new) - func(x)) < precision:
            end_time = time.time()
            print_results(x0, x_new[0], func(x_new), i + 1, end_time - start_time, "Gradient Descent")
            return x_new[0], func(x_new), i + 1
        x = x_new
    
    end_time = time.time()
    print_results(x0, x[0], func(x), max_iterations, end_time - start_time, "Gradient Descent")
    return x[0], func(x), max_iterations
