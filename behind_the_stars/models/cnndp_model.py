from keras.layers import Embedding, Conv1D, GlobalMaxPooling1D
from keras.layers import Dense, Dropout, Concatenate, Input
from keras.models import Model
from keras.callbacks import EarlyStopping
from keras.layers import SpatialDropout1D

def initialize_model(sequence_length,
                     vocab_size,
                     embedding_dim=128):
    """
    Initialize & compile CNN Model
    """

    embedding_dim = embedding_dim

    input_layer = Input(shape=(sequence_length,))
    embedding = Embedding(vocab_size, embedding_dim)(input_layer)
    embedding = SpatialDropout1D(0.4)(embedding)

    # Parallel CNN filters
    conv3 = Conv1D(128, 5, activation='relu')(embedding)
    conv4 = Conv1D(128, 7, activation='relu')(embedding)
    conv5 = Conv1D(128, 10, activation='relu')(embedding)

    pool5 = GlobalMaxPooling1D()(conv3)
    pool7 = GlobalMaxPooling1D()(conv4)
    pool10 = GlobalMaxPooling1D()(conv5)

    # Concatenate pooled features
    concat = Concatenate()([pool5, pool7, pool10])
    drop = Dropout(0.3)(concat)
    dense = Dense(128, activation='relu')(drop)
    drop2  = Dropout(0.3)(dense)
    output = Dense(1, activation='sigmoid')(drop2)
    model_CD = Model(inputs=input_layer, outputs=output)

    model_CD.compile(
        loss='binary_crossentropy',
        optimizer='adam',
        metrics=['accuracy','precision','recall']
    )

    return model_CD

def train_model(model,
                X,
                y,
                patience=2,
                epochs=15,
                validation_split=0.3,
                batch_size=100):

    """
    Fit the model
    """

    early_stop = EarlyStopping(patience=patience,
                               restore_best_weights=True)

    history = model.fit(X, y,
                               epochs=epochs,
                               validation_split=validation_split,
                               batch_size=batch_size,
                               shuffle=True,
                               callbacks=[early_stop])

    return model, history.history
