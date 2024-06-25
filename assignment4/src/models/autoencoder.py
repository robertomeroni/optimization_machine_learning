import numpy as np
import pandas as pd
from tensorflow.keras.models import Model # type: ignore
from tensorflow.keras.layers import Input, Dense # type: ignore
from tensorflow.keras.optimizers import Adam # type: ignore
import config

def create_autoencoder(input_dim, latent_dim, encoder_layers, decoder_layers, learning_rate):
    input_layer = Input(shape=(input_dim,))
    
    # Encoder
    x = input_layer
    for layer in encoder_layers:
        x = Dense(layer['units'], activation=layer['activation'])(x)
    encoder = Dense(latent_dim, activation='relu')(x)
    
    # Decoder
    x = encoder
    for layer in decoder_layers:
        x = Dense(layer['units'], activation=layer['activation'])(x)
    output_layer = Dense(input_dim, activation='linear')(x)
    
    autoencoder = Model(inputs=input_layer, outputs=output_layer)
    autoencoder.compile(optimizer=Adam(learning_rate=learning_rate), loss='mse')
    
    return autoencoder

def mean_imputation(data):
    return data.fillna(data.mean())

def autoencoder(data, latent_dim=None, epochs=None, batch_size=None, tolerance=None, learning_rate=None, encoder_layers=None, decoder_layers=None):
    ae_config = config.autoencoder_config['base']
    
    latent_dim = int(latent_dim if latent_dim is not None else ae_config['latent_dim'])
    epochs = int(epochs if epochs is not None else ae_config['epochs'])
    batch_size = int(batch_size if batch_size is not None else ae_config['batch_size'])
    tolerance = tolerance if tolerance is not None else ae_config['tolerance']
    learning_rate = learning_rate if learning_rate is not None else ae_config['learning_rate']
    encoder_layers = encoder_layers if encoder_layers is not None else ae_config['encoder_layers']
    decoder_layers = decoder_layers if decoder_layers is not None else ae_config['decoder_layers']
    
    # Mean imputation for initial missing value placeholders
    data_imputed = mean_imputation(data)
    
    input_dim = data.shape[1]
    autoencoder = create_autoencoder(input_dim, latent_dim, encoder_layers, decoder_layers, learning_rate)
    
    previous_data = data_imputed.copy()
    for epoch in range(epochs):
        autoencoder.fit(data_imputed, data_imputed, epochs=1, batch_size=batch_size, verbose=0)
        reconstructed_data = autoencoder.predict(data_imputed, verbose=0)
        
        # Ensure the shapes match before assignment
        reconstructed_data_df = pd.DataFrame(reconstructed_data, columns=data.columns)
        
        # Fill missing values with reconstructed data
        for col in data.columns:
            data_imputed.loc[data[col].isna(), col] = reconstructed_data_df.loc[data[col].isna(), col]
        
        # Check for convergence
        if np.mean(np.abs(data_imputed - previous_data)) < tolerance:
            print(f"Converged after {epoch + 1} epochs")
            break
        
        previous_data = data_imputed.copy()
    
    data_reconstructed = pd.DataFrame(data_imputed, columns=data.columns)
    return data_reconstructed
