from keras import Sequential, layers, Input, optimizers
""" in testing phase, overfitting for now """

def initialize_model_cnndp(sequence_length, vocab_size=3000) -> Sequential:
    """CNN + LSTM Architecture"""
    model = Sequential([
        Input(shape=(sequence_length,)),
        layers.Embedding(input_dim=vocab_size, output_dim=128, mask_zero=False),
        layers.Dropout(0.2),
        layers.Conv1D(64, kernel_size=5, activation='relu', padding='same'),
        layers.MaxPooling1D(pool_size=4),
        layers.LSTM(64, return_sequences=False),
        layers.Dense(32, activation='relu'),
        layers.Dropout(0.2),
        layers.Dense(1, activation='sigmoid')
    ])

    return model

def compile_model(model: Sequential, learning_rate=0.001) -> Sequential:
    """Compile the model"""
    model.compile(
        loss='binary_crossentropy',
        optimizer=optimizers.Adam(learning_rate=learning_rate),
        metrics=['accuracy']
    )
    return model
