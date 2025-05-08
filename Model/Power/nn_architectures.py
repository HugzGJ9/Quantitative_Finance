from __future__ import annotations

from keras import Sequential
from keras.models import Model
from keras.layers import Input, Dense, Dropout, BatchNormalization, LeakyReLU, LSTM


def _build_ffnn(input_dim):

    model = Sequential([
        Dense(64, activation='relu', input_shape=(input_dim,)),
        Dense(32, activation='relu'),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mse')
    return model
def _build_ffnn2(input_dim):
    inp = Input(shape=(input_dim,))
    x = Dense(128)(inp)
    x = BatchNormalization()(x)
    x = LeakyReLU()(x)
    x = Dropout(0.3)(x)

    x = Dense(64)(x)
    x = LeakyReLU()(x)
    x = Dropout(0.2)(x)

    out = Dense(1)(x)
    model = Model(inputs=inp, outputs=out)
    model.compile(optimizer="adam", loss="mse")
    return model