import numpy as np
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.kernel_ridge import KernelRidge
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import ElasticNet
from sklearn.ensemble import GradientBoostingRegressor, AdaBoostRegressor
from sklearn.tree import DecisionTreeRegressor

# Overriding the specific configurations for each model
global_config = {
    'allow_run': True,  
    'allow_plot': False,  
    'allow_model_plot': False,  
    'find_best': False, 
    'run_fss': True,
    'save_fss_data': False,
    'save_fss_plot': False,
    'run_shuffle': True,
    'save_shuffle_data': False,
    'save_actual_vs_predicted_plot': False,
    'save_errors_plot': False,
    'apply_gridsearch': False,
    'print_results': True,
    'print_best': False,
    'data_path': 'BC-Data-Set.csv',
    'allow_file_write': False,
    'allow_save': False
}

# Run configuration
run_config = {
    'Support Vector Regression': True,
    'Random Forest': True,
    'Feed-Forward Neural Network': True,
    'Gaussian Process': True,
    'Kernel Ridge Regression': True,
    'K-Nearest Neighbor': True,
    'Elastic Net Regression': True,
    'Polynomial Elastic Net Regression': True,
    'Gradient Boosting Regression': True,
    'AdaBoost Regression': True,
    'Decision Tree Regression': True
}

plot_config = {
    'Support Vector Regression': {
        'plot': True,
        'plot_params': ('C', 'gamma'),
        'plot_filter': {'kernel': 'rbf'}
    },
    'Random Forest': {
        'plot': True,
        'plot_params': ('n_estimators', 'max_depth'),
        'plot_filter': {}  # No filter, plot all data
    },
    'Feed-Forward Neural Network': {
        'plot': True,
        'plot_params': ('alpha', 'hidden_layer_sizes'),
        'plot_filter': {}  # No filter, plot all data
    },
    'Gaussian Process': {
        'plot': True,
        'plot_params': ('alpha', 'n_restarts_optimizer'),
        'plot_filter': {}  # No filter, plot all data
    },
    'Kernel Ridge Regression': {
        'plot': True,
        'plot_params': ('alpha', 'gamma'),
        'plot_filter': {}  # No filter, plot all data
    },
    'K-Nearest Neighbor': {
        'plot': True,
        'plot_params': ('n_neighbors', 'leaf_size'),
        'plot_filter': {}  # No filter, plot all data
    },
    'Elastic Net Regression': {
        'plot': True,
        'plot_params': ('alpha', 'l1_ratio'),
        'plot_filter': {}  # No filter, plot all data
    },
    'Polynomial Elastic Net Regression': {
        'plot': True,
        'plot_params': ('alpha', 'l1_ratio'),
        'plot_filter': {}  # No filter, plot all data
    },
    'Gradient Boosting Regression': {
        'plot': True,
        'plot_params': ('n_estimators', 'learning_rate'),
        'plot_filter': {}  # No filter, plot all data
    },
    'AdaBoost Regression': {
        'plot': True,
        'plot_params': ('n_estimators', 'learning_rate'),
        'plot_filter': {'loss': 'linear'}  # Filter to plot only results with loss='linear'
    },
    'Decision Tree Regression': {
        'plot': True,
        'plot_params': ('max_depth', 'min_samples_split'),
        'plot_filter': {}  # No filter, plot all data
    }
}

model_plot_config = {
    'Support Vector Regression': {
        'plot': True,
        'plot_type': 'prediction_curve',
        'params': {
            'C': 10,
            'gamma': 0.1,
            'epsilon': 0.2,
            'kernel': 'rbf'
        },
    },
    'Random Forest': {
        'plot': True,
        'plot_type': 'feature_importance',
        'params': {
            'n_estimators': 50,
            'max_depth': 30,
            'min_samples_split': 2,
            'min_samples_leaf': 2
        },
    },
    'Feed-Forward Neural Network': {
        'plot': True,
        'plot_type': 'training_curve',
        'params': {
            'hidden_layer_sizes': [200, 100],
            'activation': 'relu',
            'solver': 'adam',
            'alpha': 0.1,
            'learning_rate': 'adaptive'
        },
    },
    'Gaussian Process': {
        'plot': True,
        'plot_type': 'prediction_curve',
        'params': {
            'alpha': 0.1,
            'n_restarts_optimizer': 0
        },
    },
    'Kernel Ridge Regression': {
        'plot': True,
        'plot_type': 'prediction_curve',
        'params': {
            'alpha': 0.01,
            'kernel': 'rbf',
            'gamma': 0.01
        },
    },
    'K-Nearest Neighbor': {
        'plot': True,
        'plot_type': 'prediction_curve',
        'params': {
            'n_neighbors': 7,
            'weights': 'distance',
            'algorithm': 'auto',
            'leaf_size': 20
        },
    },
    'Elastic Net Regression': {
        'plot': True,
        'plot_type': 'prediction_curve',
        'params': {
            'alpha': 0.000001,
            'l1_ratio': 0.1,
            'max_iter': 1000
        },
    },
    'Polynomial Elastic Net Regression': {
        'plot': True,
        'plot_type': 'prediction_curve',
        'params': {
            'alpha': 0.000001,
            'l1_ratio': 0.1,
            'max_iter': 1000
        },
    },
    'Gradient Boosting Regression': {
        'plot': True,
        'plot_type': 'feature_importance',
        'params': {
            'n_estimators': 200,
            'learning_rate': 0.1,
            'max_depth': 3,
            'subsample': 0.9
        },
    },
    'AdaBoost Regression': {
        'plot': True,
        'plot_type': 'feature_importance',
        'params': {
            'n_estimators': 300,
            'learning_rate': 0.01,
            'loss': 'exponential'
        },
    },
    'Decision Tree Regression': {
        'plot': True,
        'plot_type': 'tree_structure',
        'params': {
            'max_depth': 10,
            'min_samples_split': 2,
            'min_samples_leaf': 16
        },
    }
}

model_plot_config_extended = {
    'Support Vector Regression': {
        'plot': True,
        'plot_type': 'prediction_curve',
        'params': {
            'C': [1, 10, 100, 1000],
            'gamma': [0.01, 0.1, 1, 10],
            'epsilon': [2, 4, 6, 8],
            'kernel': ['linear', 'poly', 'rbf', 'sigmoid']
        },
        'save_folder': 'fig/svr',
        'save_filename': '{kernel}_prediction_surface_C{C}_gamma{gamma}_epsilon{epsilon}.png'
    },
    'Random Forest': {
        'plot': True,
        'plot_type': 'feature_importance',
        'params': {
            'n_estimators': [2, 5, 10, 100],
            'max_depth': [1, 3, 5, None],
            'min_samples_split': [2, 10, 40, 100],
            'min_samples_leaf': [1, 10, 20, 50]
        },
        'save_folder': 'fig/rf',
        'save_filename': 'rf_feature_importance_n{n_estimators}_depth{max_depth}_split{min_samples_split}_leaf{min_samples_leaf}.png'
    },
    'Feed-Forward Neural Network': {
        'plot': True,
        'plot_type': 'training_curve',
        'params': {
            'hidden_layer_sizes': [(50,), (100,), (100, 50), (200, 100)],
            'activation': ['relu', 'tanh', 'logistic', 'identity'],
            'solver': ['adam', 'sgd', 'lbfgs'],
            'alpha': [0.0001, 0.001, 0.01, 0.1],
            'learning_rate': ['constant', 'adaptive', 'invscaling']
        },
        'save_folder': 'fig/ffnn',
        'save_filename': 'ffnn_training_curve_layers{hidden_layer_sizes}_act{activation}_solver{solver}_alpha{alpha}_lr{learning_rate}.png'
    },
    'Gaussian Process': {
        'plot': True,
        'plot_type': 'prediction_curve',
        'params': {
            'alpha': [1e-2, 1e-1, 1, 10],
            # 'n_restarts_optimizer': [0, 1, 5, 10]
        },
        'save_folder': 'fig/gp',
        'save_filename': 'gp_prediction_surface_alpha{alpha}_restarts{n_restarts_optimizer}.png'
    },
    'Kernel Ridge Regression': {
        'plot': True,
        'plot_type': 'prediction_curve',
        'params': {
            'alpha': [0.01, 0.1, 1, 10],
            'kernel': ['linear', 'poly', 'rbf', 'sigmoid'],
            'gamma': [0.01, 0.1, 1, None]
        },
        'save_folder': 'fig/krr',
        'save_filename': 'krr_prediction_surface_kernel{kernel}_alpha{alpha}_gamma{gamma}.png'
    },
    'K-Nearest Neighbor': {
        'plot': True,
        'plot_type': 'prediction_curve',
        'params': {
            'n_neighbors': [3, 5, 7, 10],
            'weights': ['uniform', 'distance'],
            'algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute'],
            'leaf_size': [20, 30, 40, 50]
        },
        'save_folder': 'fig/knn',
        'save_filename': 'knn_prediction_surface_neighbors{n_neighbors}_weights{weights}_algo{algorithm}_leaf{leaf_size}.png'
    },
    'Elastic Net Regression': {
        'plot': True,
        'plot_type': 'prediction_curve',
        'params': {
            'alpha': [0.01, 0.1, 1, 10],
            'l1_ratio': [0.0001, 0.001, 0.01, 0.1],
            'max_iter': [100, 500, 1000, 2000]
        },
        'save_folder': 'fig/en',
        'save_filename': 'en_prediction_surface_alpha{alpha}_l1{l1_ratio}_iter{max_iter}.png'
    },
    'Polynomial Elastic Net Regression': {
        'plot': True,
        'plot_type': 'prediction_curve',
        'params': {
            'alpha': [0.01, 0.1, 1, 10],
            'l1_ratio': [0.0001, 0.001, 0.01, 0.1],
            'max_iter': [100, 500, 1000, 2000]
        },
        'save_folder': 'fig/pen',
        'save_filename': 'pen_prediction_surface_alpha{alpha}_l1{l1_ratio}_iter{max_iter}.png'
    },
    'Gradient Boosting Regression': {
        'plot': True,
        'plot_type': 'feature_importance',
        'params': {
            'n_estimators': [100, 200, 300, 400],
            'learning_rate': [0.01, 0.1, 0.2, 0.3],
            'max_depth': [3, 4, 5, 6],
            'subsample': [0.1, 0.3, 0.7, 1]
        },
        'save_folder': 'fig/gbr',
        'save_filename': 'gbr_feature_importance_n{n_estimators}_lr{learning_rate}_depth{max_depth}_sub{subsample}.png'
    },
    'AdaBoost Regression': {
        'plot': True,
        'plot_type': 'feature_importance',
        'params': {
            'n_estimators': [50, 100, 200, 300],
            'learning_rate': [0.01, 0.1, 0.5, 1.0],
            'loss': ['linear', 'square', 'exponential']
        },
        'save_folder': 'fig/abr',
        'save_filename': 'abr_feature_importance_n{n_estimators}_lr{learning_rate}_loss{loss}.png'
    },
    'Decision Tree Regression': {
        'plot': True,
        'plot_type': 'tree_structure',
        'params': {
            'max_depth': [1, 3, 5, None],
            'min_samples_split': [2, 5, 10, 20],
            'min_samples_leaf': [1, 2, 4, 8]
        },
        'save_folder': 'fig/dtr',
        'save_filename': 'dtr_tree_structure_depth{max_depth}_split{min_samples_split}_leaf{min_samples_leaf}.png'
    }
}


linspace_plot_config = {
    'Support Vector Regression': {
        'params': {
            'C': np.geomspace(0.001, 1000, 10),
            'gamma': np.geomspace(0.0001, 1, 10),
        },
        'output_file': 'outputs/Support_Vector_Regression_results.json',
        'estimator': SVR(),
        'plot': True
    },
    'Random Forest': {
        'params': {
            'n_estimators': np.linspace(1, 300, 10, dtype=int),
            'max_depth': np.linspace(1, 100, 10, dtype=int)
        },
        'output_file': 'outputs/Random_Forest_results.json',
        'estimator': RandomForestRegressor(),
        'plot': True
    },
    'Feed-Forward Neural Network': {
        'params': {
            'alpha': np.geomspace(1e-10, 1, 25),
            'activation': ['identity', 'logistic', 'tanh', 'relu']
        },
        'output_file': 'outputs/Feed-Forward_Neural_Network_results.json',
        'estimator': MLPRegressor(max_iter=1000),
        'plot': True
    },
    'Gaussian Process': {
        'params': {
            'alpha': np.geomspace(1e-3, 1, 10),
            'n_restarts_optimizer': np.linspace(0, 100, 10, dtype=int)
        },
        'output_file': 'outputs/Gaussian_Process_results.json',
        'estimator': GaussianProcessRegressor(),
        'plot': True
    },
    'Kernel Ridge Regression': {
        'params': {
            'alpha': np.geomspace(1e-4, 20, 33),
            'kernel': ['linear', 'poly', 'rbf']
        },
        'output_file': 'outputs/Kernel_Ridge_Regression_results.json',
        'estimator': KernelRidge(),
        'plot': True
    },
    'K-Nearest Neighbor': {
        'params': {
            'n_neighbors': np.linspace(1, 100, 50, dtype=int),
            'weights': ['uniform', 'distance']
        },
        'output_file': 'outputs/K-Nearest_Neighbor_results.json',
        'estimator': KNeighborsRegressor(),
        'plot': True
    },
    'Elastic Net Regression': {
        'params': {
            'alpha': np.geomspace(1e-4, 10, 10),
            'l1_ratio': np.linspace(0, 1, 10)
        },
        'output_file': 'outputs/Elastic_Net_Regression_results.json',
        'estimator': ElasticNet(),
        'plot': True
    },
    'Polynomial Elastic Net Regression': {
        'params': {
            'alpha': np.geomspace(1e-4, 10, 10),
            'l1_ratio': np.linspace(0, 1, 10)
        },
        'output_file': 'outputs/Polynomial_Elastic_Net_Regression_results.json',
        'estimator': ElasticNet(),
        'plot': True
    },
    'Gradient Boosting Regression': {
        'params': {
            'n_estimators': np.linspace(1, 1000, 10, dtype=int),
            'learning_rate': np.linspace(0.05, 1, 10)
        },
        'output_file': 'outputs/Gradient_Boosting_Regression_results.json',
        'estimator': GradientBoostingRegressor(),
        'plot': True
    },
    'AdaBoost Regression': {
        'params': {
            'n_estimators': np.linspace(1, 1000, 10, dtype=int),
            'learning_rate': np.geomspace(1e-10, 1, 10)
        },
        'output_file': 'outputs/AdaBoost_Regression_results.json',
        'estimator': AdaBoostRegressor(),
        'plot': True
    },
    'Decision Tree Regression': {
        'params': {
            'max_depth': np.linspace(1, 100, 10, dtype=int),
            'min_samples_split': np.linspace(2, 100, 10, dtype=int)
        },
        'output_file': 'outputs/Decision_Tree_Regression_results.json',
        'estimator': DecisionTreeRegressor(),
        'plot': True
    }
}


test_model_config = {
    'Support Vector Regression': {
        'estimator': SVR(),
        'param_grid': {
            'kernel': ['rbf'],
            'C': [0.1, 1, 10, 100],
            'epsilon': [0.5, 1, 2, 3, 4, 5, 6],
            'gamma': [0.1, 0.01]
        },
        'apply_gridsearch': 0,
        'print_results': 0,
        'print_best': 1,
        'output_file': 'outputs/support_vector_regression_results.json'
    },
    'Random Forest': {
        'estimator': RandomForestRegressor(),
        'param_grid': {
            'n_estimators': [50, 100, 200, 300, 400, 500, 600],
            'max_depth': [None, 10, 20, 30, 40, 50, 60],
            'min_samples_split': [2, 5, 10, 15, 20],
            'min_samples_leaf': [1, 2, 4, 8, 16]
        },
        'apply_gridsearch': 0,
        'print_results': 0,
        'print_best': 0,
        'output_file': 'outputs/random_forest_results.json'
    },
    'Feed-Forward Neural Network': {
        'estimator': MLPRegressor(),
        'param_grid': {
            'hidden_layer_sizes': [(50,), (100,), (150,), (200,), (100, 100), (150, 100), (200, 100)],
            'activation': ['relu', 'tanh'],
            'solver': ['adam', 'lbfgs'],
            'alpha': [0.0001, 0.001, 0.01, 0.1],
            'learning_rate': ['constant', 'adaptive']
        },
        'apply_gridsearch': 0,
        'print_results': 0,
        'print_best': 0,
        'output_file': 'outputs/ffnn_results.json'
    },
    'Gaussian Process': {
        'estimator': GaussianProcessRegressor(),
        'param_grid': {
            'alpha': [0.01, 0.1, 0.5, 1, 3],
            'n_restarts_optimizer': [0, 1, 2, 5]
        },
        'apply_gridsearch': 0,
        'print_results': 0,
        'print_best': 0,
        'output_file': 'outputs/gaussian_process_results.json'
    },
    'Kernel Ridge Regression': {
        'estimator': KernelRidge(),
        'param_grid': {
            'alpha': [0.01, 0.1, 1, 10, 100],
            'kernel': ['linear', 'rbf'],
            'gamma': [1, 0.1, 0.01, 0.001]
        },
        'apply_gridsearch': 0,
        'print_results': 0,
        'print_best': 0,
        'output_file': 'outputs/kernel_ridge_regression_results.json'
    },
    'K-Nearest Neighbor': {
        'estimator': KNeighborsRegressor(),
        'param_grid': {
            'n_neighbors': [3, 5, 7, 9, 11, 13, 15],
            'weights': ['uniform', 'distance'],
            'algorithm': ['auto', 'ball_tree', 'kd_tree'],
            'leaf_size': [20, 30, 40, 50, 60]
        },
        'apply_gridsearch': 0,
        'print_results': 0,
        'print_best': 0,
        'output_file': 'outputs/k_nearest_neighbor_results.json'
    },
    'Elastic Net Regression': {
        'estimator': ElasticNet(),
        'param_grid': {
            'alpha': [0.00001, 0.0001, 0.001, 0.01, 0.1, 1],
            'l1_ratio': [0.0001, 0.001, 0.1, 0.3, 0.5, 0.7, 0.9],
            'max_iter': [100, 500, 1000, 2000, 3000]
        },
        'apply_gridsearch': 0,
        'print_results': 0,
        'print_best': 0,
        'output_file': 'outputs/elastic_net_results.json'
    },
    'Polynomial Elastic Net Regression': {
        'estimator': ElasticNet(),
        'param_grid': {
            'alpha': [0.00001, 0.0001, 0.001, 0.01, 0.1, 1],
            'l1_ratio': [0.0001, 0.001, 0.1, 0.3, 0.5, 0.7, 0.9],
            'max_iter': [100, 500, 1000, 2000, 3000]
        },
        'apply_gridsearch': 0,
        'print_results': 0,
        'print_best': 0,
        'output_file': 'outputs/elastic_net_results.json'
    },
    'Gradient Boosting Regression': {
        'estimator': GradientBoostingRegressor(),
        'param_grid': {
            'n_estimators': [50, 100, 200, 300, 400, 500],
            'learning_rate': [0.01, 0.05, 0.1, 0.2, 0.3],
            'max_depth': [3, 5, 7, 9],
            'subsample': [0.7, 0.8, 0.9, 1.0]
        },
        'apply_gridsearch': 0,
        'print_results': 0,
        'print_best': 0,
        'output_file': 'outputs/gradient_boosting_results.json'
    },
    'AdaBoost Regression': {
        'estimator': AdaBoostRegressor(),
        'param_grid': {
            'n_estimators': [50, 100, 200, 300],
            'learning_rate': [0.01, 0.1, 0.5, 1.0],
            'loss': ['linear', 'square', 'exponential']
        },
        'apply_gridsearch': 0,
        'print_results': 0,
        'print_best': 0,
        'output_file': 'outputs/adaboost_results.json'
    },
    'Decision Tree Regression': {
        'estimator': DecisionTreeRegressor(),
        'param_grid': {
            'max_depth': [None, 10, 20, 30, 40, 50, 60],
            'min_samples_split': [2, 5, 10, 20],
            'min_samples_leaf': [1, 2, 4, 8, 16]
        },
        'apply_gridsearch': 0,
        'print_results': 0,
        'print_best': 0,
        'output_file': 'outputs/decision_tree_results.json'
    }
}
