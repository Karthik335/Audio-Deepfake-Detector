"""
Step 2 - Feature extraction.
Converts every audio file in the dataset into a log-Mel-spectrogram (.npy).

Expected dataset layout (this is how the Fake-or-Real 'for-2sec' dataset is organised):

    <dataset_root>/
        training/   real/*.wav   fake/*.wav
        validation/ real/*.wav   fake/*.wav
        testing/    real/*.wav   fake/*.wav

If your dataset only has   <dataset_root>/real  and  <dataset_root>/fake
(no split folders), use --auto-split and a 70/15/15 split is made for you.

Usage:
    python -m src.preprocess --data-root data/for-2seconds
    python -m src.preprocess --data-root data/my_audio --auto-split
"""
import argparse
import os
import random
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

from . import config
from .features import audio_to_mel

AUDIO_EXT = {".wav", ".mp3", ".flac", ".ogg", ".m4a"}
SPLIT_ALIASES = {
    "train": ["training", "train"],
    "val": ["validation", "val", "valid", "dev"],
    "test": ["testing", "test", "eval"],
}


def list_audio(folder: Path):
    return sorted(p for p in folder.rglob("*") if p.suffix.lower() in AUDIO_EXT)


def find_split_dir(root: Path, split: str):
    for name in SPLIT_ALIASES[split]:
        if (root / name).is_dir():
            return root / name
    return None


def collect_files(root: Path, auto_split: bool, max_per_class: int = None):
    """Return a list of (path, label, split) tuples."""
    items = []
    if not auto_split:
        for split in ["train", "val", "test"]:
            split_dir = find_split_dir(root, split)
            if split_dir is None:
                raise FileNotFoundError(
                    f"No '{SPLIT_ALIASES[split][0]}' folder in {root}. "
                    "Use --auto-split if your data only has real/ and fake/ folders."
                )
            for label, cls in enumerate(config.CLASSES):
                files = list_audio(split_dir / cls)
                if max_per_class:
                    random.shuffle(files)
                    files = files[:max_per_class]
                items += [(f, label, split) for f in files]
    else:
        for label, cls in enumerate(config.CLASSES):
            files = list_audio(root / cls)
            random.shuffle(files)
            if max_per_class:
                files = files[:max_per_class]
            n = len(files)
            n_train, n_val = int(0.70 * n), int(0.15 * n)
            for i, f in enumerate(files):
                split = "train" if i < n_train else "val" if i < n_train + n_val else "test"
                items.append((f, label, split))
    return items


def _process_one(job):
    """Worker: convert one file. Returns (ok, metadata_row_or_error)."""
    path, label, split, out_path = job
    try:
        np.save(out_path, audio_to_mel(path))
    except Exception as e:  # corrupt / unreadable file
        return False, f"{Path(path).name}: {e}"
    return True, {"file": out_path.name, "source": path, "label": label, "split": split}


def main():
    ap = argparse.ArgumentParser(description="Convert audio to Mel-spectrograms")
    ap.add_argument("--data-root", required=True, help="dataset folder")
    ap.add_argument("--out-dir", default=str(config.FEATURE_DIR))
    ap.add_argument("--auto-split", action="store_true",
                    help="dataset has only real/ and fake/ folders; make a 70/15/15 split")
    ap.add_argument("--max-per-class", type=int, default=None,
                    help="limit files per class per split (useful for a quick trial run)")
    ap.add_argument("--workers", type=int, default=os.cpu_count(),
                    help="parallel processes for feature extraction")
    args = ap.parse_args()

    random.seed(config.SEED)
    root = Path(args.data_root)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    items = collect_files(root, args.auto_split, args.max_per_class)
    print(f"Found {len(items)} audio files")

    rows, failed = [], 0
    jobs = [(str(path), label, split, out_dir / f"{split}_{config.CLASSES[label]}_{idx:06d}.npy")
            for idx, (path, label, split) in enumerate(items)]
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for ok, info in tqdm(pool.map(_process_one, jobs, chunksize=32), total=len(jobs),
                             desc="Extracting Mel-spectrograms"):
            if ok:
                rows.append(info)
            else:
                failed += 1
                print(f"  skipped {info}")

    df = pd.DataFrame(rows)
    df.to_csv(out_dir / "metadata.csv", index=False)

    print(f"\nSaved {len(df)} spectrograms to {out_dir}  (skipped {failed})")
    print(df.groupby(["split", "label"]).size().unstack(fill_value=0)
            .rename(columns=dict(enumerate(config.CLASSES))))
    if len(df):
        print("Spectrogram shape:", np.load(out_dir / df.file.iloc[0]).shape)


if __name__ == "__main__":
    main()
