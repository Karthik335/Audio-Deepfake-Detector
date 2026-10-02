"""Predict a single audio file from the command line.

Usage:
    python -m src.predict path/to/clip.wav
"""
import sys

import torch

from . import config
from .features import audio_to_mel, normalize
from .model import load_model

_model = None


def get_model():
    global _model
    if _model is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        _model = load_model(config.MODEL_PATH, device)
    return _model


@torch.no_grad()
def predict_file(path_or_file):
    """Return (label, probability_fake, mel_db) for one audio file."""
    model = get_model()
    device = next(model.parameters()).device
    mel = audio_to_mel(path_or_file)
    x = torch.from_numpy(normalize(mel)).unsqueeze(0).unsqueeze(0).to(device)
    p_fake = torch.sigmoid(model(x)).item()
    label = "fake" if p_fake >= 0.5 else "real"
    return label, p_fake, mel


if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.exit("Usage: python -m src.predict <audio_file>")
    for f in sys.argv[1:]:
        label, p, _ = predict_file(f)
        print(f"{f}: {label.upper()}  (probability of AI-generated = {p * 100:.1f}%)")
