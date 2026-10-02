"""
Signal processing: raw audio -> fixed-length waveform -> log-Mel-spectrogram.
The same function is used during preprocessing AND in the web app,
so training and inference always see identical features.
"""
import numpy as np
import librosa

from . import config


def load_audio(path_or_file, sr: int = config.SAMPLE_RATE) -> np.ndarray:
    """Load any audio file (wav/mp3/flac/ogg), convert to mono, resample to `sr`."""
    y, _ = librosa.load(path_or_file, sr=sr, mono=True)
    return y


def fix_length(y: np.ndarray, num_samples: int = config.NUM_SAMPLES) -> np.ndarray:
    """Pad with zeros or truncate so every clip has exactly `num_samples` samples."""
    if len(y) >= num_samples:
        return y[:num_samples]
    return np.pad(y, (0, num_samples - len(y)), mode="constant")


def waveform_to_mel(y: np.ndarray) -> np.ndarray:
    """
    STFT -> Mel filter bank -> log (dB) scale.
    Returns a 2-D float32 array of shape (N_MELS, time_frames).
    """
    mel = librosa.feature.melspectrogram(
        y=y,
        sr=config.SAMPLE_RATE,
        n_fft=config.N_FFT,
        hop_length=config.HOP_LENGTH,
        n_mels=config.N_MELS,
        fmin=config.F_MIN,
        fmax=config.F_MAX,
        power=2.0,
    )
    mel_db = librosa.power_to_db(mel, ref=np.max)   # decibel units, max = 0 dB
    return mel_db.astype(np.float32)


def audio_to_mel(path_or_file) -> np.ndarray:
    """Full pipeline for one file: load -> fix length -> log-Mel-spectrogram."""
    y = load_audio(path_or_file)
    y = fix_length(y)
    return waveform_to_mel(y)


def normalize(mel_db: np.ndarray) -> np.ndarray:
    """Per-sample standardisation (zero mean, unit variance)."""
    return (mel_db - mel_db.mean()) / (mel_db.std() + 1e-6)
