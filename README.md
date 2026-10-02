# Audio Deepfake Detector

Detects whether a voice recording is a **real human voice** or **AI-generated (cloned / text-to-speech)**.

Audio is converted into a **log-Mel-spectrogram** (a 2-D image of frequency over time), and a **ResNet-18** convolutional neural network classifies that image as *real* or *fake*. A **FastAPI** web app lets anyone upload a clip and get the probability that it is AI-generated.

```
Audio file ─► Resample 16 kHz ─► Pad/trim to 2 s ─► STFT ─► Mel filter bank ─► log (dB)
          ─► 128 × 126 Mel-spectrogram ─► ResNet-18 ─► sigmoid ─► P(fake)
```

---

## Project structure

```
audio_deepfake_detector/
├── src/
│   ├── config.py              # all settings (sample rate, Mel bands, epochs, ...)
│   ├── features.py            # audio -> log-Mel-spectrogram (used in training AND the web app)
│   ├── preprocess.py          # Step 2: converts the whole dataset to .npy spectrograms
│   ├── dataset.py             # PyTorch Dataset + SpecAugment
│   ├── model.py               # Step 3: ResNet-18 adapted to 1-channel input, 1 output
│   ├── train.py               # Step 4: training with BCE loss, early stopping
│   ├── evaluate.py            # Step 4b: Accuracy, Precision, Recall, F1, AUC, EER + plots
│   ├── predict.py             # predict a single file from the command line
│   └── visualize_pipeline.py  # waveform -> STFT -> Mel filters -> Mel-spectrogram figure
├── app/
│   ├── main.py                # Step 5: FastAPI backend
│   └── static/index.html      # upload page
├── notebooks/
│   └── Audio_Deepfake_Detector_Colab.ipynb   # runs everything on Google Colab
├── data/                      # dataset + extracted features go here
├── outputs/                   # trained model, metrics, plots
└── requirements.txt
```

---

## Option A: Run on Google Colab (recommended, free GPU)

1. Open [colab.research.google.com](https://colab.research.google.com) → **Upload** → choose `notebooks/Audio_Deepfake_Detector_Colab.ipynb`.
2. **Runtime → Change runtime type → T4 GPU**.
3. Run the cells top to bottom. When asked, upload `audio_deepfake_detector.zip`.

The notebook downloads the dataset, extracts features, trains, evaluates, runs the web app, and lets you download the trained model and all plots.

---

## Option B: Run on your own computer

```bash
pip install -r requirements.txt
```

**1. Get the dataset.** Download the [Fake-or-Real (FoR) dataset](https://www.kaggle.com/datasets/mohammedabdeldayem/the-fake-or-real-dataset) from Kaggle and use the **`for-2seconds`** folder. It must look like:

```
for-2seconds/
    training/   real/  fake/
    validation/ real/  fake/
    testing/    real/  fake/
```

**2. Extract Mel-spectrograms**
```bash
python -m src.preprocess --data-root path/to/for-2seconds
# quick trial on a subset:
python -m src.preprocess --data-root path/to/for-2seconds --max-per-class 1000
```

**3. Train**
```bash
python -m src.train                 # uses settings in src/config.py
python -m src.train --epochs 20 --batch-size 64
```

**4. Evaluate**
```bash
python -m src.evaluate
```

**5. Predict one file**
```bash
python -m src.predict my_clip.wav
```

**6. Web app**
```bash
uvicorn app.main:app --reload
```
Open <http://127.0.0.1:8000>, upload a clip, and see the verdict, probability and spectrogram.
API docs are auto-generated at <http://127.0.0.1:8000/docs>.

**Using a different dataset:** if your data has only `real/` and `fake/` folders (no train/val/test split), add `--auto-split` to make a 70 / 15 / 15 split. If you use 4-second clips (FoR `for-norm`), set `DURATION = 4.0` in `src/config.py`.

---

## Outputs (for the project report)

| File | What it is |
|---|---|
| `outputs/best_model.pt` | trained model weights (best validation loss) |
| `outputs/training_curves.png` | loss and accuracy per epoch |
| `outputs/confusion_matrix.png` | test-set confusion matrix |
| `outputs/roc_curve.png` | ROC curve with AUC and EER marked |
| `outputs/sample_spectrograms.png` | real vs fake Mel-spectrogram side by side |
| `outputs/signal_pipeline.png` | waveform → STFT → Mel filters → Mel-spectrogram |
| `outputs/test_metrics.json` | Accuracy, Precision, Recall, F1, ROC-AUC, EER |
| `outputs/training_history.csv` | per-epoch numbers for tables |

---

## Design choices (useful for the viva)

- **Why Mel-spectrograms?** Raw waveforms are long (32,000 samples for 2 s) and noisy. A Mel-spectrogram compresses this into a 128 × 126 image that emphasises frequency bands the way human hearing does, so a standard image CNN can learn from it.
- **Why ResNet-18?** Residual (skip) connections make deeper networks train reliably. It's light enough (~11 M parameters) to train on a free Colab GPU. ImageNet pre-trained weights are used, with the first layer converted from 3 colour channels to 1 by averaging the filters.
- **Loss:** `BCEWithLogitsLoss` (binary cross-entropy) with a single output logit; class imbalance is handled with `pos_weight`.
- **Overfitting control:** SpecAugment (random time and frequency masking), dropout before the classifier, weight decay, learning-rate reduction on plateau, and early stopping.
- **EER (Equal Error Rate):** the operating point where the false-acceptance and false-rejection rates are equal. It's the standard metric in ASVspoof anti-spoofing challenges; lower is better.
- **Data split:** FoR ships with separate training, validation and testing folders, which are used as-is so test clips are never seen during training.

## Known limitations

- The model only learns the TTS systems present in the training data. Voices from newer generators it has never seen may fool it, so testing on clips from a different source (e.g. your own recordings and an online TTS tool) is a good extra experiment.
- Only the first 2 seconds of an uploaded clip are analysed.
- Real recordings with heavy compression or background noise can be flagged as fake more often.
