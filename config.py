"""
Central configuration for the Audio Deepfake Detector.
Change values here instead of editing them across files.
"""
from pathlib import Path

# ---------------- Paths ----------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"            # raw dataset goes here
FEATURE_DIR = PROJECT_ROOT / "data" / "features"   # Mel-spectrogram .npy files
OUTPUT_DIR = PROJECT_ROOT / "outputs"       # model weights, plots, metrics
MODEL_PATH = OUTPUT_DIR / "best_model.pt"

# ---------------- Audio ----------------
SAMPLE_RATE = 16000      # every clip is resampled to 16 kHz
DURATION = 2.0           # seconds; FoR "for-2sec" clips are 2 s. Use 4.0 for "for-norm"
NUM_SAMPLES = int(SAMPLE_RATE * DURATION)

# ---------------- Mel-spectrogram ----------------
N_FFT = 1024             # STFT window size
HOP_LENGTH = 256         # STFT step
N_MELS = 128             # number of Mel bands (image height)
F_MIN = 20
F_MAX = 8000             # Nyquist for 16 kHz audio

# ---------------- Labels ----------------
CLASSES = ["real", "fake"]   # 0 = real (bonafide), 1 = fake (spoof)

# ---------------- Training ----------------
BATCH_SIZE = 32
EPOCHS = 15
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-4
EARLY_STOP_PATIENCE = 4  # stop if validation loss does not improve for N epochs
NUM_WORKERS = 2
SEED = 42
USE_PRETRAINED = True    # ImageNet weights for ResNet-18 (falls back to random if download fails)
