"""
Step 4b - Evaluation on the held-out test set.
Reports Accuracy, Precision, Recall, F1, ROC-AUC and EER (Equal Error Rate,
the standard ASVspoof metric), and saves the confusion matrix, ROC curve
and sample spectrograms for the project report.

Usage:
    python -m src.evaluate
"""
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import (accuracy_score, classification_report, confusion_matrix,
                             f1_score, precision_score, recall_score, roc_auc_score, roc_curve)
from torch.utils.data import DataLoader

from . import config
from .dataset import MelDataset
from .model import load_model


def compute_eer(y_true, scores):
    """EER = the point where False Acceptance Rate equals False Rejection Rate."""
    fpr, tpr, thresholds = roc_curve(y_true, scores)
    fnr = 1 - tpr
    idx = np.nanargmin(np.abs(fnr - fpr))
    eer = (fpr[idx] + fnr[idx]) / 2
    return float(eer), float(thresholds[idx])


@torch.no_grad()
def predict(model, loader, device):
    probs, labels = [], []
    for x, y in loader:
        probs.append(torch.sigmoid(model(x.to(device))).cpu().numpy())
        labels.append(y.numpy())
    return np.concatenate(labels).astype(int), np.concatenate(probs)


def plot_confusion(cm, path):
    fig, ax = plt.subplots(figsize=(4.5, 4))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks([0, 1], ["Real", "Fake"]); ax.set_yticks([0, 1], ["Real", "Fake"])
    ax.set(xlabel="Predicted", ylabel="Actual", title="Confusion Matrix (Test Set)")
    for i in range(2):
        for j in range(2):
            ax.text(j, i, cm[i, j], ha="center", va="center", fontsize=14,
                    color="white" if cm[i, j] > cm.max() / 2 else "black")
    fig.colorbar(im, ax=ax, fraction=0.046); fig.tight_layout()
    fig.savefig(path, dpi=150); plt.close(fig)


def plot_roc(y_true, scores, auc, eer, path):
    fpr, tpr, _ = roc_curve(y_true, scores)
    fig, ax = plt.subplots(figsize=(5, 4.5))
    ax.plot(fpr, tpr, lw=2, label=f"ResNet-18 (AUC = {auc:.4f})")
    ax.plot([0, 1], [0, 1], "--", color="grey", label="Random")
    ax.plot([eer], [1 - eer], "ro", label=f"EER = {eer * 100:.2f}%")
    ax.set(xlabel="False Positive Rate", ylabel="True Positive Rate", title="ROC Curve (Test Set)")
    ax.legend(loc="lower right"); ax.grid(alpha=.3); fig.tight_layout()
    fig.savefig(path, dpi=150); plt.close(fig)


def plot_samples(ds, path):
    """Side-by-side Mel-spectrograms of a real and a fake clip (for the report)."""
    fig, ax = plt.subplots(1, 2, figsize=(11, 3.8))
    for k, (lab, name) in enumerate([(0, "Real (bonafide)"), (1, "Fake (AI-generated)")]):
        row = ds.df[ds.df.label == lab].iloc[0]
        mel = np.load(ds.feature_dir / row.file)
        im = ax[k].imshow(mel, origin="lower", aspect="auto", cmap="magma")
        ax[k].set(title=name, xlabel="Time frames", ylabel="Mel bands")
        fig.colorbar(im, ax=ax[k], format="%+2.0f dB")
    fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = load_model(config.MODEL_PATH, device)
    test_ds = MelDataset("test")
    loader = DataLoader(test_ds, batch_size=config.BATCH_SIZE, shuffle=False)

    y_true, scores = predict(model, loader, device)
    y_pred = (scores >= 0.5).astype(int)
    eer, eer_thr = compute_eer(y_true, scores)
    auc = roc_auc_score(y_true, scores)
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])

    metrics = {
        "test_samples": int(len(y_true)),
        "accuracy": accuracy_score(y_true, y_pred),
        "precision_fake": precision_score(y_true, y_pred, zero_division=0),
        "recall_fake": recall_score(y_true, y_pred, zero_division=0),
        "f1_fake": f1_score(y_true, y_pred, zero_division=0),
        "roc_auc": auc,
        "eer_percent": eer * 100,
        "eer_threshold": eer_thr,
        "confusion_matrix": cm.tolist(),
    }
    print("\n========== TEST RESULTS ==========")
    for k, v in metrics.items():
        if k != "confusion_matrix":
            print(f"{k:>16}: {v:.4f}" if isinstance(v, float) else f"{k:>16}: {v}")
    print("\n" + classification_report(y_true, y_pred, target_names=["Real", "Fake"], digits=4))

    config.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(config.OUTPUT_DIR / "test_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2, default=float)
    plot_confusion(cm, config.OUTPUT_DIR / "confusion_matrix.png")
    plot_roc(y_true, scores, auc, eer, config.OUTPUT_DIR / "roc_curve.png")
    plot_samples(test_ds, config.OUTPUT_DIR / "sample_spectrograms.png")
    print(f"Plots and metrics saved to {config.OUTPUT_DIR}")


if __name__ == "__main__":
    main()
