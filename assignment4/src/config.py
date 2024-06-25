# config.py

global_config = {
    'data_file': 'data_matrix.csv',
    'missing_percentage': 0.1,  
    'burst_length': 2,    
    'set_seed': True,
    'allow_save_output': True,
    'allow_save_metrics_plot': True,
    'allow_run_models': True,
}

run_model = {
    'polynomial_interpolation': True,
    'lstm': True,
    'mice_mlr': True,
    'mice_knn': True,
    'autoencoder': True,
    'dlin': True
}


lstm_config = {
    'base': {
        'look_back': 3,
        'epochs': 20,
        'batch_size': 32,
        'num_layers': 1,
        'learning_rate': 0.0075,
        'dropout_rate': 0.2,
        'hidden_units': 8,
        'activation_function': 'relu',
        'optimizer': 'adam',
        'verbose': 0
    }
}



mice_config = {
    'mlr': {
        'base': {
            'max_iter': 20,
            'random_state': 25,
        }
    },
    'knn': {
        'base': {
            'n_neighbors': 20,
            'weights': 'uniform',  # Can be 'uniform' or 'distance'
            'metric': 'nan_euclidean'  # Can be 'nan_euclidean' or 'nan_manhattan'
        }
    }
}

autoencoder_config = {
    'base': {
        'latent_dim': 3,
        'epochs': 25,
        'batch_size': 32,
        'tolerance': 1e-5,
        'learning_rate': 0.0001,
        'encoder_layers': [
            {'units': 64, 'activation': 'relu'},
            {'units': 32, 'activation': 'relu'},
            # {'units': 16, 'activation': 'relu'},
            # {'units': 8, 'activation': 'relu'}
        ],
        'decoder_layers': [
            # {'units': 8, 'activation': 'relu'},
            # {'units': 16, 'activation': 'relu'},
            {'units': 32, 'activation': 'relu'},
            {'units': 64, 'activation': 'relu'}
        ]
    }
}


