# Two-Stream Contextual Fusion for Real-Time Driver Safety Alert

**Author:** Rzhevskaia · Student ID: 2540041027  
**Course:** Data Mining  
**Presentation:** 1 June 2026

---

## Project Description

A two-stream deep learning system that fuses road scene danger classification (Stream 1, BDD100K) with in-cabin driver attention monitoring (Stream 2, Drive&Act / MediaPipe) into a single probabilistic safety alert.

Key novelties:
- **OHEM loss** — Online Hard Example Mining for class imbalance
- **Scale-Invariant Skeletal Normalization** — camera-agnostic pose features
- **Soft Voting Fusion** — `risk = 0.6 × danger_norm + 0.4 × (1 − attention_norm)`, alert threshold 0.65

---

## Setup Instructions

### Environment

Python 3.10+ and Google Colab (recommended — the notebook uses Colab-specific cells for Google Drive mounting and webcam capture).

### Install Dependencies

```bash
pip install -r requirements.txt
```

> **MediaPipe note:** Version 0.10+ removes `mp.solutions`. The notebook uses the new `mediapipe.tasks.python.vision` API with downloaded `.task` model files. Do **not** install mediapipe < 0.10.13 (unavailable on PyPI).

### Dataset Access

| Dataset | Size | Download |
|---------|------|----------|
| BDD100K images (100k split) | ~6.5 GB | https://bdd-data.berkeley.edu/ |
| BDD100K labels | ~130 MB | included with BDD100K download |
| Drive&Act (inner_mirror) | ~varies | https://driveandact.com/ |

or you can download these datasets from my path https://drive.google.com/drive/folders/1aXhR0l7nVHss_gWqcISjxxwro7jM6-s0?usp=sharing

Place datasets on Google Drive under `MyDrive/data mining/`:
- `archive (3).zip` — BDD100K images + labels
- `inner_mirror.zip` — Drive&Act cabin video


---

## Run Instructions

1. Open `danger_detection_week13_final.ipynb` in Google Colab.
2. Mount Google Drive and place datasets as described above.
3. Run cells **in order** from top to bottom.
4. MediaPipe `.task` model files are downloaded automatically on first run.

### Key sections

| Section | Content |
|---------|---------|
| A | 4-class BDD100K labelling + tf.data pipeline |
| B | 3-block CNN (baseline) |
| C | ResNet-50 fine-tuning (3-phase) |
| D | Drive&Act annotation parsing |
| E | MediaPipe attention pipeline |
| F | Soft Voting fusion demo |
| G | OHEM loss implementation + comparison |
| H | Scale-Invariant Skeletal Normalization |
| J | Live webcam demo (Colab only) |

---

## Expected Output

- `outputs_week13/best_cnn_4class.keras` — trained CNN weights
- `outputs_week13/best_resnet50_4class.keras` — trained ResNet-50 weights
- `outputs_week13/drive_act_labeled.csv` — annotated Drive&Act frames
- `outputs_week13/stream2_eval.csv` — Stream 2 classification report
- `outputs_week13/*.png` — training curves, confusion matrices, fusion chart

---

## AI Tool Acknowledgement

Claude (Anthropic) was used as a coding assistant for debugging MediaPipe API migration (v0.9 → v0.10), OHEM loss implementation, and code structure. All design decisions and written content are the author's own work.
