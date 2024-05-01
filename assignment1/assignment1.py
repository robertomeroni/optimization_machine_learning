import cvx_check
import algorithms
import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize as opt
import numdifftools as nd
import cvxpy as cp



def exercise_1():
    # Define the objective function
    def objective_function(x):
        x1, x2 = x[0], x[1]
        return np.exp (x1) * (4 * x1**2 + 2 * x2**2 + 4 * x1 * x2 + 2 * x2 + 1)

    # Constraints
    def constraint1(x):
        return x[0] * x[1] - x[0] - x[1] + 1.5

    def constraint2(x):
        return -x[0] * x[1] - 10

    
    ## a
    cvx_check.is_psd([objective_function])

    ## b
    constraints = [
        {'type': 'ineq', 'fun': lambda x: -constraint1(x)},
        {'type': 'ineq', 'fun': lambda x: -constraint2(x)}
    ]

    initial_guesses = [np.array([0, 0]), np.array([10, 20]), np.array([-10, 1]), np.array([-30, -30])]

    timings_no_jacobian = cvx_check.optimize_with_timing(initial_guesses, objective_function, constraints)

    ## c
    def jacobian(x):
        x1, x2 = x
        df_dx1 = (8 * x1 + 4 * x2) * np.exp(x1) + (4 * x1**2 + 4 * x1 * x2 + 2 * x2**2 + 2 * x2 + 1) * np.exp(x1)
        df_dx2 = (4 * x1 + 4 * x2 + 2) * np.exp(x1)
        return np.array([df_dx1, df_dx2])
    
    timings_with_jacobian = cvx_check.optimize_with_timing(initial_guesses, objective_function, constraints, jacobian)

    for i in range(len(initial_guesses)):
        print(f"Difference in timings for initial guess {initial_guesses[i]}: {timings_with_jacobian[i] - timings_no_jacobian[i]}")

def exercise_2():
    # Define the objective function
    def objective_function(x):
        return x[0]**2 + x[1]**2

    # Constraints
    def constraint1(x):
        return -x[0] + 0.5  # Corresponds to 0.5 ≤ x1

    def constraint2(x):
        return -x[0] - x[1] + 1  # Corresponds to −x1 − x2 + 1 ≤ 0

    def constraint3(x):
        return -x[0]**2 - x[1]**2 + 1  # Corresponds to −x1^2 − x2^2 + 1 ≤ 0

    def constraint4(x):
        return -9*x[0]**2 - x[1]**2 + 9  # Corresponds to −9x1^2 − x2^2 + 9 ≤ 0

    def constraint5(x):
        return -x[0]**2 + x[1]  # Corresponds to −x1^2 + x2 ≤ 0

    def constraint6(x):
        return -x[1]**2 + x[0]  # Corresponds to −x2^2 + x1 ≤ 0


    
    ## a
    cvx_check.is_psd([objective_function])
    # cvx_check.plot_feasible_region([constraint1, constraint2, constraint3, constraint4, constraint5, constraint6])

    ## b
    constraints = [{'type': 'ineq', 'fun': lambda x: -constraint1(x)},
                   {'type': 'ineq', 'fun': lambda x: -constraint2(x)},
                   {'type': 'ineq', 'fun': lambda x: -constraint3(x)},
                   {'type': 'ineq', 'fun': lambda x: -constraint4(x)},
                   {'type': 'ineq', 'fun': lambda x: -constraint5(x)},
                   {'type': 'ineq', 'fun': lambda x: -constraint6(x)}]

    initial_guesses = [np.array([5, 5]), np.array([5, -2.5]), np.array([5, -7.5]), np.array([-100, -100])]

    timings_no_jacobian = cvx_check.optimize_with_timing(initial_guesses, objective_function, constraints)

    ## c
    def jacobian(x):
        x1, x2 = x
        df_dx1 = 2 * x1
        df_dx2 = 2 * x2
        return np.array([df_dx1, df_dx2])
    
    timings_with_jacobian = cvx_check.optimize_with_timing(initial_guesses, objective_function, constraints, jacobian)

    for i in range(len(initial_guesses)):
        print(f"Time difference for initial guess {initial_guesses[i]}: {timings_with_jacobian[i] - timings_no_jacobian[i]}")

def exercise_3():
    # Define the objective function
    def objective_function(x):
        return x[0]**2 + x[1]**2

    # Constraints
    def constraint1(x):
        return x[0]**2 + x[0]*x[1] + x[1]**2 - 3

    def constraint2(x):
        return -(3*x[0] + 2*x[1] - 3)


    
    ## a
    cvx_check.is_psd([objective_function])
    cvx_check.plot_feasible_region([constraint1, constraint2])

    ## b
    constraints = [{'type': 'ineq', 'fun': lambda x: -constraint1(x)},
                   {'type': 'ineq', 'fun': lambda x: -constraint2(x)},]

    initial_guesses = [np.array([5000, 5000]), np.array([5, -2.5]), np.array([5, -7.5]), np.array([-100000000, 100000])]

    timings_no_jacobian = cvx_check.optimize_with_timing(initial_guesses, objective_function, constraints)

    ## c

    # Define the variables
    x1 = cp.Variable()
    x2 = cp.Variable()

    # Define the objective
    objective = cp.Minimize(cp.square(x1) + cp.square(x2))

    # Define the constraints
    z = (x1 + 0.5*x2)
    constraints = [cp.square(z) + 0.75*cp.square(x2) <= 3, 3*x1 + 2*x2 >= 3]

    # Define the problem and solve it
    problem = cp.Problem(objective, constraints)
    problem.solve()

    # Output the solution
    print(f"Solution: x1 = {x1.value}, x2 = {x2.value}")
    print(f"Objective value: {problem.value}")

    # Dual variables (Lagrange multipliers for the constraints)
    print(f"Dual variables: {constraints[0].dual_value}, {constraints[1].dual_value}")

def exercise_4():
    # Define the objective function
    def objective_function(x):
        return x[0]**2 + 1

    # Constraints
    def constraint1(x):
        return (x[0] - 2)*(x[0] - 4) 
    
    ## a

    # Define the variables
    x = cp.Variable()

    # Define the objective
    objective = cp.Minimize(cp.square(x) + 1)

    # Define the constraints
    constraints = [cp.square(x) - 6*x + 8 <= 0]

    # Define the problem and solve it
    problem = cp.Problem(objective, constraints)
    problem.solve()

    # Output the solution
    print(f"Solution: x = {x.value}")
    print(f"Objective value: {problem.value}")
    
    # Dual variables (Lagrange multipliers for the constraints)
    print(f"Dual variables: {constraints[0].dual_value}")

    ## b
    # Define the dual variable 
    lambda_var = cp.Variable(nonneg=True)

    # Define the objective
    objective = cp.Maximize((-cp.square(lambda_var) + 9*lambda_var + 1) / (lambda_var + 1))

    # Define the constraints
    constraints = [lambda_var >= 0]

    # Define the problem
    dual_problem = cp.Problem(objective, constraints)

    # Solve the problem
    dual_problem.solve(qcp=True)

    print("Optimal lambda:", lambda_var.value)
    print("Optimal dual value:", dual_problem.value)

def exercise_5():
    def objective_function(x):
        return x[0]**2 + x[1]**2
    
    def constraint1(x):
        return (x[0] - 1)**2 + (x[1] - 1)**2 - 1
    
    def constraint2(x):
        return (x[0] - 1)**2 + (x[1] + 1)**2 - 1
    
    cvx_check.plot_feasible_region([constraint1, constraint2])
    cvx_check.is_psd([objective_function])

    ## a
    # Define variables
    x1 = cp.Variable()
    x2 = cp.Variable()

    # Objective function
    objective = cp.Minimize(x1**2 + x2**2)

    # Constraints
    constraints = [
        cp.square(x1 - 1) + cp.square(x2 - 1) <= 1,
        cp.square(x1 - 1) + cp.square(x2 + 1) <= 1,
    ]

    # Define problem
    problem = cp.Problem(objective, constraints)

    # Solve problem
    problem.solve()

    # Display results
    print("Optimal values:")
    print(f"x1 = {x1.value}")
    print(f"x2 = {x2.value}")
    print(f"Minimum objective value = {problem.value}")
    print(f"Constraint 1 dual value = {constraints[0].dual_value}")
    print(f"Constraint 2 dual value = {constraints[1].dual_value}")

    ## b
    # Define the dual variables
    lambda1 = cp.Variable(nonneg=True)
    lambda2 = cp.Variable(nonneg=True)

    # Define the objective function to minimize (negative of q(lambda))
    objective = cp.Maximize(( - cp.square(lambda1 - lambda2) + lambda1 + lambda2 ) / (lambda1 + lambda2 + 1))

    # Define the constraints
    constraints = [lambda1 >= 0, lambda2 >= 0]

    # Define the problem
    problem = cp.Problem(objective, constraints)

    # Solve the problem
    problem.solve(qcp=True)

    # Print the solution
    print("Lambda 1:", lambda1.value)
    print("Lambda 2:", lambda2.value)
    print("Optimal dual value:", problem.value)

def exercise_6():
    ## Solve with custom function
    # Example 1
    func1 = lambda x: 2*x**2 - 0.5
    x0_1 = 3
    solution1 = 0

    result1_gd = algorithms.gradient_descent(func1, x0_1)
    result1_nm = algorithms.newton_method(func1, x0_1)

    # Example 2
    func2 = lambda x: 2*x**4 - 4*x**2 + x - 0.5
    initial_points = [-2, -0.5, 0.5, 2]
    solution2 = -1.05745377

    result2_gd = [algorithms.gradient_descent(func2, x0) for x0 in initial_points]
    result2_nm = [algorithms.newton_method(func2, x0) for x0 in initial_points]
    
    print("Iterations example 1:")
    print("Gradient descent:", result1_gd[2])
    print("Newton's method:", result1_nm[2])

    print("Iterations example 2:")
    print("Gradient descent:")
    for i, res in enumerate(result2_gd):
        print(f"Initial point {initial_points[i]}: {res[2]}")

    print("Newton's method:")
    for i, res in enumerate(result2_nm):
        print(f"Initial point {initial_points[i]}: {res[2]}")

    print("Accuracy example 1:")
    print("Gradient descent:", np.abs(result1_gd[0] - solution1))
    print("Newton's method:", np.abs(result1_nm[0] - solution1))

    print("Accuracy example 2:")
    print("Gradient descent:")
    for i, res in enumerate(result2_gd):
        print(f"Initial point {initial_points[i]}: {np.abs(res[0] - solution2)}")

    print("Newton's method:")
    for i, res in enumerate(result2_nm):
        print(f"Initial point {initial_points[i]}: {np.abs(res[0] - solution2)}")

def exercise_7():
    ## a
    # Define the capacities of the links
    C = np.array([1, 2, 1, 2, 1])

    # Define the matrix A based on the routes of each source
    A = np.array([[1, 0, 1],  
                  [1, 1, 0],  
                  [0, 0, 0],
                  [0, 0, 0],
                  [0, 0, 1]]) 

    # Number of sources
    num_sources = A.shape[1]

    # Decision variables for the rates of each source
    x = cp.Variable(num_sources)

    # Objective function: Sum of utilities of all sources
    objective = cp.Maximize(cp.sum(cp.log(x)))

    # Constraints: Ax <= C and x >= 0
    constraints = [A @ x <= C, x >= 0]

    # Problem definition
    problem = cp.Problem(objective, constraints)

    # Solve the problem
    problem.solve()

    # Results
    x_optimal = x.value
    lagrange_multipliers = constraints[0].dual_value

    print("Optimal rates for each source:", x_optimal)
    print("Optimal value of the objective function:", problem.value)
    print("Lagrange multipliers (dual solution) for each link:", lagrange_multipliers)



def exercise_8():
    # Define the variables
    x = cp.Variable(3) # Users 1, 2, and 3
    T = cp.Variable(3) # Time fractions for the links

    # Define the objective
    objective = cp.Maximize(cp.sum(cp.log(x)))

    # Define the constraints
    constraints = [
        x[0] + x[1] <= T[0], # x1 + x2 ≤ T12
        x[0] <= T[1],       # x1 ≤ T23
        x[2] <= T[2],       # x3 ≤ T32
        T[0] + T[1] + T[2] <= 1, # T12 + T23 + T32 ≤ 1
        x >= 0, # Traffic allocated must be non-negative
        T >= 0, # Time fractions must be non-negative
    ]

    # Define the problem and solve
    problem = cp.Problem(objective, constraints)
    problem.solve()

    # Display the results
    print(f"Optimal traffic allocation (x): {x.value}")
    print(f"Optimal time fractions (T): {T.value}")
    print(f"Dual variables:")
    for i, constraint in enumerate(constraints):
        print(f"Constraint {i+1} dual variable: {constraint.dual_value}")



exercise_8()