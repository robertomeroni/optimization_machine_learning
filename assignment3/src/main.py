import data_processing as dp
import parameter_tuning as pt
from model_config import global_config, run_config, plot_config, linspace_plot_config
import os
import time

def run_model(model_name, X_train, X_test, y_train, y_test, scaler_y):
    config = linspace_plot_config[model_name]

    if model_name == 'Polynomial Elastic Net Regression':
        X_train, X_test = dp.get_polynomial_features(X_train, X_test, degree=2)

    results = pt.parameter_tuning(
        model_name, 
        config['estimator'], 
        config['params'], 
        X_train, 
        X_test, 
        y_train, 
        y_test, 
        scaler_y, 
        apply_gridsearch=global_config.get('apply_gridsearch', False), 
        print_results=global_config.get('print_results', False),
        print_best=False,  
        output_file=config.get('output_file'),  
        allow_file_write=global_config.get('allow_file_write', False)
    )
    return results


def find_best_model():
    if global_config.get('find_best', False):
        model_names = list(linspace_plot_config.keys())
        for model_name in model_names:
            output_file = linspace_plot_config[model_name]['output_file']
            if os.path.exists(output_file):
                pt.find_best_models_from_file(output_file)

def plot_r2():
    if global_config.get('allow_plot', True):
        for model_name, config in plot_config.items():
            if config.get('plot', False):
                dp.plot_r2_scores(model_name, linspace_plot_config[model_name]['output_file'])

def run_fss(X, y):
    fss_results = {}
    if global_config.get('run_fss', False):
        for model_name, config in linspace_plot_config.items():
            file_path = config['output_file']
            print(f"Processing {model_name} with results from {file_path}")
            selected_features, scores = pt.from_best_model_to_fss(model_name, file_path, X, y)
            fss_results[model_name] = {
                'selected_features': selected_features,
                'scores': scores
            }
            print(f"Selected Features for {model_name}: {selected_features}")
            print(f"Scores for each step for {model_name}: {scores}")
            print()
    return fss_results

def run_plot_fss():
        for model_name, config in plot_config.items():
            json_file = f"outputs/fss/fss_results_{model_name.replace(' ', '_')}.json"
            if os.path.exists(json_file):
                dp.plot_fss_r2_scores(model_name, json_file)
            else:
                print(f"File {json_file} does not exist. Skipping plotting for {model_name}.")

def run_shuffle_and_evaluate(X, y, num_shuffles=10):
    shuffle_results = {}
    if global_config.get('run_shuffle', False):
        for model_name, config in linspace_plot_config.items():
            json_file = config['output_file']
            shuffle_stats = dp.shuffle_and_evaluate(model_name, json_file, X, y, num_shuffles=num_shuffles)
            shuffle_results[model_name] = shuffle_stats
            print(f"Shuffle and evaluate for {model_name} completed.")
            print()
    return shuffle_results

def run_plot_model_errors(X_train, X_test, y_train, y_test, scaler_y):
    for model_name, config in linspace_plot_config.items():
        file_path = config['output_file']
        print(f"Plotting errors for {model_name} using results from {file_path}")
        dp.plot_model_errors(model_name, file_path, X_train, X_test, y_train, y_test, scaler_y)

def run_plot_actual_vs_predicted(X_train, X_test, y_train, y_test, scaler_y):
    for model_name, config in linspace_plot_config.items():
        file_path = config['output_file']
        print(f"Plotting actual vs predicted for {model_name} using results from {file_path}")
        dp.plot_actual_vs_predicted(model_name, file_path, X_train, X_test, y_train, y_test, scaler_y)



data, X, y = dp.data_extraction(global_config['data_path'])
X_train, X_test, y_train, y_test, scaler_X, scaler_y = dp.split_and_standardize(X, y)

# Get model names from the run_config dictionary
if global_config.get('allow_run', True):
    model_names = [name for name, should_run in run_config.items() if should_run]
else:
    model_names = []

print(f"Running the following models: {model_names}\n")

# Run the models
for model_name in model_names:
    start_time = time.time()  # Start timing
    results = run_model(model_name, X_train, X_test, y_train, y_test, scaler_y)
    end_time = time.time()  # End timing
    
    elapsed_time = end_time - start_time
    print(f"Time taken to run {model_name}: {elapsed_time:.2f} seconds")
    if global_config.get('allow_file_write', False):
        print(f"Results stored in {linspace_plot_config[model_name]['output_file']}\n")



# dp.plot_all_models_performances(linspace_plot_config)
dp.create_and_plot_models(X, y)
# fss_results = run_fss(X, y)
# run_plot_fss()
# run_plot_model_errors(X_train, X_test, y_train, y_test, scaler_y)
# run_plot_actual_vs_predicted(X_train, X_test, y_train, y_test, scaler_y)
# shuffle_results = run_shuffle_and_evaluate(X, y, num_shuffles=10)
r2_scores = pt.get_all_r2_scores()