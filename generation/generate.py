import numpy as np
from tensorflow.keras.models import load_model
from preprocessing.preprocess import load_and_clean_text, tokenize
from utils.config import MODEL_PATH, SEQ_LENGTH

def generate_text(seed, length=300, temperature=0.8):
    text = load_and_clean_text()
    _, char_to_idx, idx_to_char, vocab_size = tokenize(text)

    model = load_model(MODEL_PATH)
    generated = seed.lower()

    for _ in range(length):
        seq = [char_to_idx[c] for c in generated[-SEQ_LENGTH:]]
        seq = np.array(seq).reshape(1, -1)

        preds = model.predict(seq, verbose=0)[0]
        preds = np.log(preds + 1e-8) / temperature
        probs = np.exp(preds) / np.sum(np.exp(preds))

        next_idx = np.random.choice(len(probs), p=probs)
        generated += idx_to_char[next_idx]

    return generated

if __name__ == "__main__":
    print(generate_text("to be or not to be"))
