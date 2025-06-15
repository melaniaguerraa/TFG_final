from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, GRU, Dense, Dropout, LSTM, Input
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam
import tensorflow as tf

def build_cnn_gru_model(input_shape, forecast_horizon=24):
    model = Sequential([
        Input(shape=input_shape),
        Conv1D(filters=64, kernel_size=3, activation='relu'),
        GRU(128, return_sequences=False),
        Dense(32, activation='relu'),
        Dropout(0.3),
        Dense(forecast_horizon),  # 24 h output 
    ])
    
    model.compile(loss='mae', optimizer=Adam(learning_rate=0.001), metrics=['mse'])
    return model

## LSTM
def build_lstm(input_shape, forecast_horizon=24):
    model = Sequential([
        Input(shape=input_shape),
        LSTM(128, return_sequences=True),
        Dropout(0.2),
        LSTM(64, return_sequences=True), 
        Dropout(0.2),
        LSTM(32),
        Dense(forecast_horizon)
    ])
    model.compile(loss='mae', optimizer=Adam(learning_rate=0.001), metrics=['mse'])
    return model

def fit_cnn_gru_model(model, X_train_seq, y_train_seq, X_val_seq, y_val_seq, epochs=50, batch_size=32):
    early_stopping = EarlyStopping(monitor='val_loss', patience=7, restore_best_weights=True)
    history = model.fit(
        X_train_seq, y_train_seq, 
        epochs=epochs, 
        batch_size=batch_size,
        verbose=1, callbacks=[early_stopping], 
        validation_data=(X_val_seq, y_val_seq))
    return history


