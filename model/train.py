import os
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

from preprocessing.preprocess import (
    load_and_clean_text,
    tokenize,
    create_sequences
)
from model.model import build_model
from utils.config import MODEL_PATH, EPOCHS, BATCH_SIZE

def train():
    text = load_and_clean_text()
    encoded, char_to_idx, idx_to_char, vocab_size = tokenize(text)
    X, y = create_sequences(encoded)

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.1
    )

    model = build_model(vocab_size)

    os.makedirs("model", exist_ok=True)

    callbacks = [
        EarlyStopping(patience=3, restore_best_weights=True),
        ModelCheckpoint(MODEL_PATH, save_best_only=True)
    ]

    model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        callbacks=callbacks
    )

    print("Training complete")

if __name__ == "__main__":
    train()
