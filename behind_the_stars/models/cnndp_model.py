import numpy as np
from keras import Sequential, layers, Input
from keras.callbacks import EarlyStopping

def initialize_model(input_shape):
    """
    Initialize & compile CNN Model
    """
    model = Sequential([
        Input(shape=input_shape),
        layers.Embedding(input_dim=5000, output_dim=128, mask_zero=False),
        layers.Dropout(0.2),
        layers.Conv1D(64, 5, activation='relu', padding='same'),
        layers.MaxPooling1D(pool_size=4),
        layers.LSTM(64, return_sequences=False),
        layers.Dense(32, activation='relu'),
        layers.Dropout(0.2),
        layers.Dense(1, activation='sigmoid')
    ])

    model.compile(
        loss='binary_crossentropy',
        optimizer='adam',
        metrics=['accuracy']
    )

    return model

def train_model(model, X, y):
    """
    Fit the model
    """

    early_stop = EarlyStopping(
        patience=2,
        restore_best_weights=True,
        verbose=0
    )

    history = model.fit(X, y, epochs=15, validation_split=0.3, batch_size=100,
        shuffle=True,
        callbacks=[early_stop],
        verbose=0
    )

    return model, history.history
