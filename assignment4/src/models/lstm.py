import pandas as pd
import numpy as np
from data_processing import ensure_dir
from tensorflow.keras.models import Sequential # type: ignore
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input # type: ignore
from tensorflow.keras.optimizers import Adam, SGD, RMSprop # type: ignore
import config



def create_lstm_model(look_back, num_layers, hidden_units, activation_function, dropout_rate, optimizer, learning_rate):
    model = Sequential()
    model.add(Input(shape=(look_back, 1)))  
    for i in range(num_layers):
        return_sequences = i < num_layers - 1
        model.add(LSTM(hidden_units, activation=activation_function, return_sequences=return_sequences))
        if dropout_rate > 0:
            model.add(Dropout(dropout_rate))
    model.add(Dense(1))
    
    if optimizer == 'adam':
        optimizer = Adam(learning_rate=learning_rate)
    elif optimizer == 'sgd':
        optimizer = SGD(learning_rate=learning_rate)
    elif optimizer == 'rmsprop':
        optimizer = RMSprop(learning_rate=learning_rate)
    
    model.compile(optimizer=optimizer, loss='mse')
    return model

def prepare_lstm_data(time_series, look_back):
    X, y = [], []
    for i in range(len(time_series) - look_back):
        X.append(time_series[i:(i + look_back)])
        y.append(time_series[i + look_back])
    return np.array(X), np.array(y)

def lstm(data_miss, look_back=None, epochs=None, batch_size=None, num_layers=None, learning_rate=None, dropout_rate=None, hidden_units=None, activation_function=None, optimizer=None, verbose=None):
    base_config = config.lstm_config['base']

    # Use provided parameters or fallback to base config
    look_back = int(look_back if look_back is not None else base_config['look_back'])
    epochs = int(epochs if epochs is not None else base_config['epochs'])
    batch_size = int(batch_size if batch_size is not None else base_config['batch_size'])
    num_layers = int(num_layers if num_layers is not None else base_config['num_layers'])
    learning_rate = float(learning_rate if learning_rate is not None else base_config['learning_rate'])
    dropout_rate = float(dropout_rate if dropout_rate is not None else base_config['dropout_rate'])
    hidden_units = int(hidden_units if hidden_units is not None else base_config['hidden_units'])
    activation_function = activation_function if activation_function is not None else base_config['activation_function']
    optimizer = optimizer if optimizer is not None else base_config['optimizer']
    verbose = int(verbose if verbose is not None else base_config['verbose'])

    filled_data = data_miss.copy()
    missing_indexes = {}

    for column in data_miss.columns:
        time_series = data_miss[column].values
        valid_indexes = np.where(~np.isnan(time_series))[0]
        miss_indexes = np.where(np.isnan(time_series))[0]
        missing_indexes[column] = miss_indexes

        # Prepare the data for LSTM
        X, y = prepare_lstm_data(time_series[valid_indexes], look_back)
        X = X.reshape((X.shape[0], X.shape[1], 1))

        # Create and train the LSTM model
        model = create_lstm_model(look_back, num_layers, hidden_units, activation_function, dropout_rate, optimizer, learning_rate)
        history = model.fit(X, y, epochs=epochs, batch_size=batch_size, verbose=verbose)

        # Predict missing values
        for idx in miss_indexes:
            if idx < look_back:
                valid_data = pd.Series(time_series[:idx]).dropna()
                if len(valid_data) > 1:
                    interpolated_value = np.interp(idx, valid_data.index, valid_data.values)
                    time_series[idx] = interpolated_value
                else:
                    time_series[idx] = valid_data.mean() if len(valid_data) == 1 else 0
            else:
                X_pred = time_series[idx-look_back:idx].reshape((1, look_back, 1))
                if not np.isnan(X_pred).any():
                    y_pred = model.predict(X_pred, verbose=0)
                    time_series[idx] = y_pred.flatten()
                else:
                    print(f"Skipping prediction for index {idx} due to NaNs in the input sequence")

        filled_data[column] = time_series

    return filled_data