"""
Draws the signal-processing pipeline for one audio file:
raw waveform -> STFT spectrogram -> Mel filter bank -> log-Mel-spectrogram.
Useful as the 'methodology' figure in the project report.

Usage:
    python -m src.visualize_pipeline path/to/clip.wav
"""
import sys

import librosa
import librosa.display
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from . import config
from .features import fix_length, load_audio, waveform_to_mel


def plot_pipeline(path, out_path=config.OUTPUT_DIR / "signal_pipeline.png"):
    y = fix_length(load_audio(path))
    stft_db = librosa.amplitude_to_db(np.abs(librosa.stft(y, n_fft=config.N_FFT,
                                                           hop_length=config.HOP_LENGTH)), ref=np.max)
    mel_fb = librosa.filters.mel(sr=config.SAMPLE_RATE, n_fft=config.N_FFT, n_mels=config.N_MELS,
                                 fmin=config.F_MIN, fmax=config.F_MAX)
    mel_db = waveform_to_mel(y)

    fig, ax = plt.subplots(4, 1, figsize=(10, 12))
    librosa.display.waveshow(y, sr=config.SAMPLE_RATE, ax=ax[0])
    ax[0].set(title="1. Raw waveform (1-D signal, 16 kHz)", xlabel="Time (s)", ylabel="Amplitude")

    img = librosa.display.specshow(stft_db, sr=config.SAMPLE_RATE, hop_length=config.HOP_LENGTH,
                                   x_axis="time", y_axis="hz", ax=ax[1])
    ax[1].set(title="2. STFT spectrogram (linear frequency, dB)")
    fig.colorbar(img, ax=ax[1], format="%+2.0f dB")

    freqs = librosa.fft_frequencies(sr=config.SAMPLE_RATE, n_fft=config.N_FFT)
    for k in range(0, config.N_MELS, 8):          # every 8th filter, for clarity
        ax[2].plot(freqs, mel_fb[k] / mel_fb[k].max(), lw=1)
    ax[2].set(title=f"3. Mel filter bank ({config.N_MELS} filters, every 8th shown): "
                    "narrow at low Hz, wide at high Hz",
              xlabel="Frequency (Hz)", ylabel="Filter weight", xlim=(0, config.F_MAX))

    img = librosa.display.specshow(mel_db, sr=config.SAMPLE_RATE, hop_length=config.HOP_LENGTH,
                                   x_axis="time", y_axis="mel", fmax=config.F_MAX, ax=ax[3], cmap="magma")
    ax[3].set(title=f"4. Log-Mel-spectrogram -> CNN input ({mel_db.shape[0]} x {mel_db.shape[1]})")
    fig.colorbar(img, ax=ax[3], format="%+2.0f dB")

    fig.tight_layout()
    config.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Saved {out_path}")
    return out_path


if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit("Usage: python -m src.visualize_pipeline <audio_file>")
    plot_pipeline(sys.argv[1])
