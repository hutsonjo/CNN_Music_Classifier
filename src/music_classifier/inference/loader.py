from __future__ import annotations

from pathlib import Path
from tensorflow.keras.models import load_model
from tensorflow.keras import Model

def load_genre_model() -> Model:
    path = Path("model/model.keras")
    if not path.exists():
        raise FileNotFoundError(f"Model file not found: {path}")
    return load_model(path)
