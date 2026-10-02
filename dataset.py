"""
PyTorch Dataset that reads the saved Mel-spectrograms.
SpecAugment (random time and frequency masking) is applied to training data
to reduce overfitting.
"""
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from . import config
from .features import normalize


def spec_augment(mel: torch.Tensor, freq_mask: int = 15, time_mask: int = 20,
                 num_masks: int = 2) -> torch.Tensor:
    """Zero out random horizontal (frequency) and vertical (time) stripes."""
    mel = mel.clone()
    _, n_mels, n_frames = mel.shape
    for _ in range(num_masks):
        f = np.random.randint(0, freq_mask + 1)
        f0 = np.random.randint(0, max(1, n_mels - f))
        mel[:, f0:f0 + f, :] = 0
        t = np.random.randint(0, time_mask + 1)
        t0 = np.random.randint(0, max(1, n_frames - t))
        mel[:, :, t0:t0 + t] = 0
    return mel


class MelDataset(Dataset):
    def __init__(self, split: str, feature_dir=config.FEATURE_DIR, augment: bool = False):
        self.feature_dir = Path(feature_dir)
        meta = pd.read_csv(self.feature_dir / "metadata.csv")
        self.df = meta[meta.split == split].reset_index(drop=True)
        self.augment = augment
        if len(self.df) == 0:
            raise ValueError(f"No samples found for split '{split}'")

    def __len__(self):
        return len(self.df)

    def __getitem__(self, i):
        row = self.df.iloc[i]
        mel = np.load(self.feature_dir / row.file)
        mel = torch.from_numpy(normalize(mel)).unsqueeze(0)   # (1, n_mels, frames)
        if self.augment:
            mel = spec_augment(mel)
        label = torch.tensor(row.label, dtype=torch.float32)
        return mel, label

    def class_counts(self):
        return self.df.label.value_counts().sort_index().tolist()
