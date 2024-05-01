import cvxpy as cp
import numpy as np

import numpy as np

def laplacian_to_adjacency(laplacian):
    threshold = 0.01
    n = laplacian.shape[0]
    adjacency_matrix = -laplacian
    np.fill_diagonal(adjacency_matrix, 0)
    adjacency_matrix[adjacency_matrix < threshold] = 0

    return adjacency_matrix


def update_L(Y, alpha, beta, m):
    L = cp.Variable((m, m), symmetric=True)
    constraints = [
        cp.sum(L, axis=1) == 0,  
        cp.trace(L) == m  
    ]
    # Ensuring off-diagonal elements are non-positive and diagonal elements are non-negative
    for i in range(m):
        for j in range(m):
            if i != j:
                constraints.append(L[i, j] <= 0)
    
    objective = cp.Minimize(alpha * cp.trace(Y.T @ L @ Y) + beta * cp.norm(L, 'fro')**2)
    problem = cp.Problem(objective, constraints)
    problem.solve(solver=cp.SCS)
    return L.value

def update_Y(X, L, alpha):
    m, n = X.shape  # m is features, n is observations
    Y = cp.Variable((m, n))
    # Apply quadratic form individually to each column of Y and sum the results
    quadratic_forms = sum(cp.quad_form(Y[:, i], L) for i in range(n))
    objective = cp.Minimize(cp.norm(X - Y, 'fro')**2 + alpha * quadratic_forms)
    problem = cp.Problem(objective)
    problem.solve(solver=cp.SCS)
    return Y.value


import numpy as np

def laplacian_model(X, alpha=0.0001, beta=0.1, max_iter=1000, tol=1e-9):
    m, n = X.shape
    L = np.zeros((m, m))  # Initial L
    Y = X.copy()  # Start with Y = X initially

    for i in range(max_iter):
        L_prev = L.copy()
        Y_prev = Y.copy()

        L = update_L(Y, alpha, beta, m)
        Y = update_Y(X, L, alpha)

        # Check for convergence
        delta_L = np.linalg.norm(L - L_prev)
        delta_Y = np.linalg.norm(Y - Y_prev)

        print(f"Iteration {i+1}: ΔL = {delta_L:.13f}, ΔY = {delta_Y:.13f}")

        if delta_L < tol and delta_Y < tol:
            print("Convergence reached.")
            break

    adjacency_matrix = laplacian_to_adjacency(L)
    return adjacency_matrix



def is_psd(mat):
    return np.all(np.linalg.eigvals(mat) >= 0)

