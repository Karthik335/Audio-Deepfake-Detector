"""
Step 4 - Training and validation.
Binary Cross-Entropy (with logits) loss, AdamW optimiser, learning-rate
scheduling, early stopping, and the best model saved by validation loss.

Usage:
    python -m src.train
    python -m src.train --epochs 20 --batch-size 64
"""
import argparse
import json
import random
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from . import config
from .dataset import MelDataset
from .model import DeepfakeResNet, count_parameters


def set_seed(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def run_epoch(model, loader, criterion, device, optimizer=None):
    """One pass over the data. Trains if an optimizer is given, otherwise evaluates."""
    train = optimizer is not None
    model.train(train)
    total_loss, correct, n = 0.0, 0, 0
    with torch.set_grad_enabled(train):
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            loss = criterion(logits, y)
            if train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            total_loss += loss.item() * len(y)
            correct += ((torch.sigmoid(logits) >= 0.5).float() == y).sum().item()
            n += len(y)
    return total_loss / n, correct / n


def plot_history(hist: pd.DataFrame, path):
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    ax[0].plot(hist.epoch, hist.train_loss, "o-", label="Train")
    ax[0].plot(hist.epoch, hist.val_loss, "s-", label="Validation")
    ax[0].set(title="Loss (BCE)", xlabel="Epoch", ylabel="Loss"); ax[0].legend(); ax[0].grid(alpha=.3)
    ax[1].plot(hist.epoch, hist.train_acc, "o-", label="Train")
    ax[1].plot(hist.epoch, hist.val_acc, "s-", label="Validation")
    ax[1].set(title="Accuracy", xlabel="Epoch", ylabel="Accuracy"); ax[1].legend(); ax[1].grid(alpha=.3)
    fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=config.EPOCHS)
    ap.add_argument("--batch-size", type=int, default=config.BATCH_SIZE)
    ap.add_argument("--lr", type=float, default=config.LEARNING_RATE)
    ap.add_argument("--no-pretrained", action="store_true")
    ap.add_argument("--no-augment", action="store_true")
    args = ap.parse_args()

    set_seed(config.SEED)
    config.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    train_ds = MelDataset("train", augment=not args.no_augment)
    val_ds = MelDataset("val")
    print(f"Train: {len(train_ds)} (real/fake = {train_ds.class_counts()})  "
          f"Val: {len(val_ds)} (real/fake = {val_ds.class_counts()})")

    train_dl = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                          num_workers=config.NUM_WORKERS, pin_memory=device == "cuda")
    val_dl = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                        num_workers=config.NUM_WORKERS)

    model = DeepfakeResNet(pretrained=config.USE_PRETRAINED and not args.no_pretrained).to(device)
    print(f"Trainable parameters: {count_parameters(model):,}")

    # Weight the loss if classes are imbalanced (pos_weight = #real / #fake)
    n_real, n_fake = train_ds.class_counts()
    pos_weight = torch.tensor([n_real / max(n_fake, 1)], device=device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=config.WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=2)

    history, best_val, patience = [], float("inf"), 0
    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        tr_loss, tr_acc = run_epoch(model, train_dl, criterion, device, optimizer)
        va_loss, va_acc = run_epoch(model, val_dl, criterion, device)
        scheduler.step(va_loss)
        history.append(dict(epoch=epoch, train_loss=tr_loss, train_acc=tr_acc,
                            val_loss=va_loss, val_acc=va_acc,
                            lr=optimizer.param_groups[0]["lr"]))
        flag = ""
        if va_loss < best_val:
            best_val, patience, flag = va_loss, 0, "  <- best, saved"
            torch.save({"model_state": model.state_dict(), "epoch": epoch,
                        "val_loss": va_loss, "val_acc": va_acc,
                        "config": {k: getattr(config, k) for k in
                                   ["SAMPLE_RATE", "DURATION", "N_FFT", "HOP_LENGTH", "N_MELS"]}},
                       config.MODEL_PATH)
        else:
            patience += 1
        print(f"Epoch {epoch:02d}/{args.epochs} | train loss {tr_loss:.4f} acc {tr_acc:.4f} | "
              f"val loss {va_loss:.4f} acc {va_acc:.4f} | {time.time() - t0:.0f}s{flag}")
        if patience >= config.EARLY_STOP_PATIENCE:
            print(f"Early stopping: no improvement for {patience} epochs.")
            break

    hist = pd.DataFrame(history)
    hist.to_csv(config.OUTPUT_DIR / "training_history.csv", index=False)
    plot_history(hist, config.OUTPUT_DIR / "training_curves.png")
    with open(config.OUTPUT_DIR / "train_summary.json", "w") as f:
        json.dump({"best_val_loss": best_val, "epochs_run": len(hist),
                   "best_val_acc": float(hist.loc[hist.val_loss.idxmin(), "val_acc"])}, f, indent=2)
    print(f"\nBest model saved to {config.MODEL_PATH}")


if __name__ == "__main__":
    main()
