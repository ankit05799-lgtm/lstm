from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense
from utils.config import EMBEDDING_DIM, LSTM_UNITS

def build_model(vocab_size):
    model = Sequential([
        Embedding(vocab_size, EMBEDDING_DIM),
        LSTM(LSTM_UNITS, return_sequences=True),
        LSTM(LSTM_UNITS),
        Dense(vocab_size, activation="softmax")
    ])

    model.compile(
        loss="sparse_categorical_crossentropy",
        optimizer="adam"
    )
    return model
