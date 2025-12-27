import numpy as np
import re
from utils.config import DATA_PATH, SEQ_LENGTH

def load_and_clean_text():
    with open(DATA_PATH, "r", encoding="utf-8") as f:
        text = f.read().lower()

    text = re.sub(r"[^a-z\s]", "", text)
    text = re.sub(r"\s+", " ", text)
    return text

def tokenize(text):
    chars = sorted(set(text))
    char_to_idx = {c: i for i, c in enumerate(chars)}
    idx_to_char = {i: c for i, c in enumerate(chars)}
    encoded = np.array([char_to_idx[c] for c in text])
    return encoded, char_to_idx, idx_to_char, len(chars)

def create_sequences(encoded):
    X, y = [], []
    for i in range(len(encoded) - SEQ_LENGTH):
        X.append(encoded[i:i + SEQ_LENGTH])
        y.append(encoded[i + SEQ_LENGTH])
    return np.array(X), np.array(y)
