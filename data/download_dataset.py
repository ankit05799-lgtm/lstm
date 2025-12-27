import requests
import os
from utils.config import DATA_URL, DATA_PATH

def download_dataset():
    os.makedirs("data", exist_ok=True)

    if not os.path.exists(DATA_PATH):
        print("Downloading Shakespeare dataset...")
        text = requests.get(DATA_URL).text
        with open(DATA_PATH, "w", encoding="utf-8") as f:
            f.write(text)
        print("Download complete")
    else:
        print("Dataset already exists")

if __name__ == "__main__":
    download_dataset()
